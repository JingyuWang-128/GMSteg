import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.optim as optim
from diffusers import AutoencoderKL
from tqdm import tqdm

from config import cfg
from models import GenMambaINN
from utils.logger import get_logger
from utils.attacks import simulate_vae_attack
from utils.datasets import get_dataloader

def set_freeze(model, frozen: bool):
    for param in model.parameters():
        param.requires_grad = not frozen

def load_vae(device):
    print(f"Loading VAE from {cfg.VAE_ID}...")
    try:
        vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device)
    except:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

def images_to_latents(vae, images):
    with torch.no_grad():
        dist = vae.encode(images).latent_dist
        # 【关键】训练时移除缩放因子，让数值范围变大，避免梯度消失
        # latents = dist.sample() * cfg.VAE_SCALE_FACTOR 
        latents = dist.sample()
    return latents

def train():
    logger = get_logger("GenMamba-Train", cfg.LOG_DIR)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = cfg.DEVICE
    
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    
    train_loader = get_dataloader(cfg.DIV2K_TRAIN_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=True)
    if train_loader is None: return

    logger.info(f"开始训练 (Total Epochs: {cfg.EPOCHS})")
    
    stage1_end = cfg.EPOCHS_STAGE1
    stage2_end = cfg.EPOCHS_STAGE1 + cfg.EPOCHS_STAGE2
    
    for epoch in range(cfg.EPOCHS):
        current_epoch = epoch + 1
        model.train()
        
        # 阶段调度
        if current_epoch <= stage1_end:
            stage = "Stage 1 (INN Warm-up)"
            set_freeze(model.inn_blocks, False) 
            set_freeze(model.rectifier, True)   
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, 0.0
            noise_std = 0.0 
        elif current_epoch <= stage2_end:
            stage = "Stage 2 (Rect Training)"
            set_freeze(model.inn_blocks, True)  
            set_freeze(model.rectifier, False)  
            w_sec, w_stego, w_rect = 0.0, 0.0, cfg.LAMBDA_RECT
            noise_std = 0.1 
        else:
            stage = "Stage 3 (Joint)"
            set_freeze(model.inn_blocks, False)
            set_freeze(model.rectifier, False)
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, cfg.LAMBDA_RECT
            noise_std = 0.1

        # 统计变量
        epoch_stats = {'all': 0.0, 'sec': 0.0, 'stg': 0.0, 'rect': 0.0, 'diff': 0.0}
        num_batches = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {current_epoch}", unit="batch")

        for step, (images, _) in pbar:
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            split = images.shape[0] // 2
            cover = images_to_latents(vae, images[:split])
            secret = images_to_latents(vae, images[split : split*2])
            
            stego_latent = model.embed(cover, secret)
            
            if noise_std > 0:
                stego_damaged = simulate_vae_attack(stego_latent, noise_std=noise_std)
            else:
                stego_damaged = stego_latent 
            
            recovered_all, z_rectified = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
            loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
            
            total_loss = (w_sec * loss_secret + w_stego * loss_stego + w_rect * loss_rect)
            
            # 监控指标
            with torch.no_grad():
                abs_diff = torch.mean(torch.abs(stego_latent[:, :4] - cover))

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 累加
            epoch_stats['all'] += total_loss.item()
            epoch_stats['sec'] += loss_secret.item()
            epoch_stats['stg'] += loss_stego.item()
            epoch_stats['rect'] += loss_rect.item()
            epoch_stats['diff'] += abs_diff.item()
            num_batches += 1

            pbar.set_postfix({'L': f"{total_loss.item():.4f}", 'Diff': f"{abs_diff.item():.4f}"})

        # 打印日志
        if num_batches > 0:
            avgs = {k: v / num_batches for k, v in epoch_stats.items()}
            logger.info(f"End Ep[{current_epoch}] {stage} | "
                        f"All: {avgs['all']:.5f} | "
                        f"Sec: {avgs['sec']:.5f} | "
                        f"Stg: {avgs['stg']:.5f} | "
                        f"Rect: {avgs['rect']:.5f} | "
                        f"Diff: {avgs['diff']:.5f}")

        if current_epoch in [stage1_end, stage2_end, cfg.EPOCHS]:
            save_name = f"model_stage_{current_epoch}.pth"
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, save_name))
            logger.info(f"Checkpoint saved: {save_name}")

if __name__ == "__main__":
    train()