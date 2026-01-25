# train.py (更新版)
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.optim as optim
from diffusers import AutoencoderKL

from config import cfg
from models import GenMambaINN
from utils.logger import get_logger
from utils.attacks import simulate_vae_attack
from utils.datasets import get_dataloader  # 使用我们新写的加载器

# --- 辅助函数 ---
def load_vae(device):
    print(f"正在加载 VAE: {cfg.VAE_ID}...")
    # 优先加载本地，失败尝试联网
    try:
        vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device)
    except:
        # 尝试使用镜像站
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device)
        
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

def images_to_latents(vae, images):
    with torch.no_grad():
        dist = vae.encode(images).latent_dist
        latents = dist.sample() * cfg.VAE_SCALE_FACTOR
    return latents

def validate(model, vae, val_loader, device, logger, epoch):
    """ 使用 DIV2K Valid 集进行验证 """
    model.eval()
    total_loss = 0
    total_sec_loss = 0
    
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            # 切分 Cover/Secret
            split = images.shape[0] // 2
            cover = images_to_latents(vae, images[:split])
            secret = images_to_latents(vae, images[split:split*2])
            
            # 推理
            stego = model.embed(cover, secret)
            # 验证时依然加入一点噪声测试鲁棒性
            stego_damaged = simulate_vae_attack(stego, noise_std=0.05)
            recovered_all, _ = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # 计算 Loss 指标
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            loss_stego = torch.mean((stego[:, :4] - cover) ** 2)
            
            total_loss += (loss_secret + loss_stego).item()
            total_sec_loss += loss_secret.item()
            
            if i >= 10: break # 验证集只跑一部分节省时间
            
    avg_loss = total_loss / (i + 1)
    avg_sec = total_sec_loss / (i + 1)
    logger.info(f"--- [Validation Epoch {epoch}] Avg Loss: {avg_loss:.6f} | Secret MSE: {avg_sec:.6f} ---")
    model.train()

def train():
    logger = get_logger("GenMamba-Train", cfg.LOG_DIR)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = cfg.DEVICE
    
    # 1. 加载模型
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    
    # 2. 加载数据集
    logger.info(f"加载训练集 DIV2K: {cfg.DIV2K_TRAIN_PATH}")
    train_loader = get_dataloader(cfg.DIV2K_TRAIN_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=True)
    
    logger.info(f"加载验证集 DIV2K Valid: {cfg.DIV2K_VALID_PATH}")
    val_loader = get_dataloader(cfg.DIV2K_VALID_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=False)

    # 3. 训练循环
    global_step = 0
    for epoch in range(cfg.EPOCHS):
        model.train()
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            split = images.shape[0] // 2
            cover_img = images[:split]
            secret_img = images[split:split*2]
            
            # Forward
            cover = images_to_latents(vae, cover_img)
            secret = images_to_latents(vae, secret_img)
            
            stego_latent = model.embed(cover, secret)
            stego_damaged = simulate_vae_attack(stego_latent, noise_std=0.1)
            
            recovered_all, z_rectified = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # Loss
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
            loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
            
            total_loss = (cfg.LAMBDA_SECRET * loss_secret + 
                          cfg.LAMBDA_STEGO * loss_stego + 
                          cfg.LAMBDA_RECT * loss_rect)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            global_step += 1
            if global_step % 20 == 0:
                logger.info(f"Ep[{epoch+1}] Step[{step}] L_all:{total_loss:.4f} L_sec:{loss_secret:.4f} L_rect:{loss_rect:.4f}")
        
        # 每个 Epoch 结束后验证
        validate(model, vae, val_loader, device, logger, epoch+1)
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()