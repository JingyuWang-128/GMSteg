# train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import AutoencoderKL

from config import cfg
from models import GenMambaINN
from utils.logger import get_logger
from utils.attacks import simulate_vae_attack

# --- VAE 辅助函数 (创新点1 基础) ---
def load_vae(device):
    print(f"正在加载预训练 VAE: {cfg.VAE_ID} ...")
    try:
        vae = AutoencoderKL.from_pretrained(cfg.VAE_ID).to(device)
    except:
        print("无法从 HuggingFace 加载，尝试加载本地缓存或请检查网络。")
        # 如果无法联网，请确保已有本地模型
        raise RuntimeError("VAE 模型加载失败")
        
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False # 冻结 VAE
    return vae

def images_to_latents(vae, images):
    """ 将 RGB 图片编码为潜码，并进行标准缩放 """
    with torch.no_grad():
        # Encode 得到分布
        dist = vae.encode(images).latent_dist
        # 采样并缩放 (SD 标准操作)
        latents = dist.sample() * cfg.VAE_SCALE_FACTOR
    return latents

# --- 数据加载 ---
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # 归一化到 [-1, 1]
    ])
    
    if not os.path.exists(cfg.DATASET_PATH):
        print(f"⚠️ 警告: 数据集路径 {cfg.DATASET_PATH} 不存在！")
        return None
        
    dataset = datasets.ImageFolder(root=cfg.DATASET_PATH, transform=transform)
    return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
                      num_workers=cfg.NUM_WORKERS, drop_last=True)

# --- 训练主程序 ---
def train():
    logger = get_logger("GenMamba-Train", cfg.LOG_DIR)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    
    # 1. 准备模型与环境
    device = cfg.DEVICE
    vae = load_vae(device)
    
    model = GenMambaINN(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LR)
    
    dataloader = get_dataloader()
    if dataloader is None:
        logger.error("无可用数据，训练终止。请在 config.py 设置正确的 DATASET_PATH")
        return

    logger.info(">>> 开始训练 GenMamba-INN (包含所有创新点)...")
    
    global_step = 0
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        
        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # 数据切分: 一半做 Cover，一半做 Secret
            B = images.shape[0]
            if B < 2: continue # 防止 Batch 太小
            
            split = B // 2
            cover_img = images[:split]
            secret_img = images[split : split*2]
            
            # --- Step A: 进入潜空间 (Innovation 1) ---
            cover = images_to_latents(vae, cover_img)
            secret = images_to_latents(vae, secret_img)
            
            # --- Step B: 频率自适应嵌入 (Innovation 2) ---
            # model.embed 内部会自动调用 FrequencyGatedSSM
            stego_latent = model.embed(cover, secret)
            
            # --- Step C: 模拟攻击与漂移 (Innovation 3 Trigger) ---
            # 必须引入攻击，LDR 模块才能学到东西
            stego_damaged = simulate_vae_attack(stego_latent, noise_std=0.1)
            
            # --- Step D: 矫正与提取 (Innovation 3 & 1) ---
            # 1. LDR 模块尝试修复 stego_damaged -> z_rectified
            # 2. INN 逆向提取 -> rec_secret
            recovered_all, z_rectified = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # --- Step E: 计算损失 ---
            # 1. 秘密信息恢复 loss
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            # 2. 隐写不可感知 loss (Latent 层面的约束)
            loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
            # 3. 漂移矫正 loss (关键: 监督 LDR 模块)
            loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
            
            total_loss = (cfg.LAMBDA_SECRET * loss_secret + 
                          cfg.LAMBDA_STEGO * loss_stego + 
                          cfg.LAMBDA_RECT * loss_rect)
            
            # --- Step F: 反向传播 ---
            optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪 (稳定 Mamba 训练)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            global_step += 1
            
            if global_step % 10 == 0:
                logger.info(f"Epoch[{epoch+1}/{cfg.EPOCHS}] Step[{step}] "
                            f"Loss: {total_loss.item():.4f} | "
                            f"Sec: {loss_secret.item():.4f} | "
                            f"Rect: {loss_rect.item():.4f}")
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            save_path = f"{cfg.CHECKPOINT_DIR}/epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"模型保存至: {save_path}")

if __name__ == "__main__":
    train()