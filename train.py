# train.py
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

# --- 辅助函数: 冻结/解冻参数 ---
def set_freeze(model, frozen: bool):
    """ 冻结或解冻模型的梯度更新 """
    for param in model.parameters():
        param.requires_grad = not frozen

# --- 辅助函数: VAE 加载 ---
def load_vae(device):
    print(f"Loading VAE from {cfg.VAE_ID}...")
    try:
        vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device)
    except:
        # Fallback to mirror if local fails
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

def images_to_latents(vae, images):
    with torch.no_grad():
        dist = vae.encode(images).latent_dist
        latents = dist.sample() * cfg.VAE_SCALE_FACTOR
    return latents

def train():
    logger = get_logger("GenMamba-Train", cfg.LOG_DIR)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = cfg.DEVICE
    
    # 1. 初始化
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    
    # 数据加载
    train_loader = get_dataloader(cfg.DIV2K_TRAIN_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=True)
    if train_loader is None: return

    logger.info(f"开始分阶段训练 (Total Epochs: {cfg.EPOCHS})")
    logger.info(f"Plan: Stage1(INN)={cfg.EPOCHS_STAGE1} -> Stage2(Rect)={cfg.EPOCHS_STAGE2} -> Stage3(Joint)={cfg.EPOCHS_STAGE3}")

    global_step = 0
    
    # 定义阶段边界
    stage1_end = cfg.EPOCHS_STAGE1
    stage2_end = cfg.EPOCHS_STAGE1 + cfg.EPOCHS_STAGE2
    
    for epoch in range(cfg.EPOCHS):
        current_epoch = epoch + 1
        model.train()
        
        # === 动态阶段调度 ===
        if current_epoch <= stage1_end:
            stage = "Stage 1 (INN Warm-up)"
            short_stage = "S1-INN"
            set_freeze(model.inn_blocks, False) 
            set_freeze(model.rectifier, True)   
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, 0.0
            noise_std = 0.0 
            
        elif current_epoch <= stage2_end:
            stage = "Stage 2 (Rectifier Training)"
            short_stage = "S2-Rect"
            set_freeze(model.inn_blocks, True)  
            set_freeze(model.rectifier, False)  
            w_sec, w_stego, w_rect = 0.0, 0.0, cfg.LAMBDA_RECT
            noise_std = 0.1 
            
        else:
            stage = "Stage 3 (Joint Fine-tuning)"
            short_stage = "S3-Joint"
            set_freeze(model.inn_blocks, False)
            set_freeze(model.rectifier, False)
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, cfg.LAMBDA_RECT
            noise_std = 0.1
            
        # 移除了此处 "进入 Stage" 的频繁打印，保持日志整洁

        # === 初始化 Epoch 统计变量 ===
        epoch_loss_all = 0.0
        epoch_loss_sec = 0.0
        epoch_loss_rect = 0.0
        num_batches = 0

        # === Batch 循环 ===
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {current_epoch} | {short_stage}", unit="batch")

        for step, (images, _) in pbar:
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            split = images.shape[0] // 2
            cover_img = images[:split]
            secret_img = images[split : split*2]
            
            # 1. 潜空间投影
            cover = images_to_latents(vae, cover_img)
            secret = images_to_latents(vae, secret_img)
            
            # 2. 嵌入
            stego_latent = model.embed(cover, secret)
            
            # 3. 模拟攻击
            if noise_std > 0:
                stego_damaged = simulate_vae_attack(stego_latent, noise_std=noise_std)
            else:
                stego_damaged = stego_latent 
            
            # 4. 提取与矫正
            recovered_all, z_rectified = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # 5. Loss 计算
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
            loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
            
            total_loss = (w_sec * loss_secret + 
                          w_stego * loss_stego + 
                          w_rect * loss_rect)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            global_step += 1
            
            # === 累加 Loss (使用 .item() 避免显存泄漏) ===
            epoch_loss_all += total_loss.item()
            epoch_loss_sec += loss_secret.item()
            epoch_loss_rect += loss_rect.item()
            num_batches += 1

            # 进度条实时显示 (可选，方便看一眼是不是跑飞了)
            pbar.set_postfix({
                'L': f"{total_loss.item():.4f}", 
                'Stg': short_stage
            })
            
            # --- 移除了原来的 global_step % 50 打印 ---

        # === Epoch 结束: 打印平均 Loss ===
        avg_loss_all = epoch_loss_all / num_batches
        avg_loss_sec = epoch_loss_sec / num_batches
        avg_loss_rect = epoch_loss_rect / num_batches

        logger.info(f"End Ep[{current_epoch}] {stage} | "
                    f"Avg Loss: {avg_loss_all:.5f} | "
                    f"Sec: {avg_loss_sec:.5f} | "
                    f"Rect: {avg_loss_rect:.5f}")

        # === 保存策略: 仅在阶段结束时保存 ===
        if current_epoch in [stage1_end, stage2_end, cfg.EPOCHS]:
            save_name = f"model_{short_stage}.pth"
            save_path = os.path.join(cfg.CHECKPOINT_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            logger.info(f"✅ 阶段完成，Checkpoint 已保存: {save_name}")
        
        # --- 移除了原来的 if current_epoch % 10 == 0 保存 ---

if __name__ == "__main__":
    train()