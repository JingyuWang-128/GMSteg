# train.py
import os
import torch
import torch.optim as optim
from diffusers import AutoencoderKL

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
    
    # 优化器: 包含所有参数 (PyTorch 会自动忽略 requires_grad=False 的参数)
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
        
        # === 动态阶段调度 (Stage Scheduling) ===
        if current_epoch <= stage1_end:
            stage = "Stage 1 (INN Warm-up)"
            # 策略: 训练 INN，冻结 Rectifier，无噪声
            set_freeze(model.inn_blocks, False) # Train INN
            set_freeze(model.rectifier, True)   # Freeze Rect
            
            # Loss权重: 只关注隐写和恢复，不管矫正
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, 0.0
            noise_std = 0.0 # 理想环境
            
        elif current_epoch <= stage2_end:
            stage = "Stage 2 (Rectifier Training)"
            # 策略: 冻结 INN，训练 Rectifier，加噪声
            set_freeze(model.inn_blocks, True)  # Freeze INN (保持隐写分布不变)
            set_freeze(model.rectifier, False)  # Train Rect
            
            # Loss权重: 只关注矫正效果 (Denoising)
            # 虽然 L_sec 也能算，但 Stage 2 核心是修补潜码，L_rect 是最直接的监督信号
            w_sec, w_stego, w_rect = 0.0, 0.0, cfg.LAMBDA_RECT
            noise_std = 0.1 # 模拟攻击环境
            
        else:
            stage = "Stage 3 (Joint Fine-tuning)"
            # 策略: 全部解冻，加噪声，全Loss
            set_freeze(model.inn_blocks, False)
            set_freeze(model.rectifier, False)
            
            w_sec, w_stego, w_rect = cfg.LAMBDA_SECRET, cfg.LAMBDA_STEGO, cfg.LAMBDA_RECT
            noise_std = 0.1
            
        # 在每个 Epoch 开始时打印当前策略
        if global_step % len(train_loader) == 0:
            logger.info(f"\n>>> 进入 {stage} | Noise: {noise_std} | Weights: Sec={w_sec} Stego={w_stego} Rect={w_rect}")

        # === Batch 循环 ===
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            split = images.shape[0] // 2
            cover_img = images[:split]
            secret_img = images[split : split*2]
            
            # 1. 潜空间投影
            cover = images_to_latents(vae, cover_img)
            secret = images_to_latents(vae, secret_img)
            
            # 2. 嵌入 (Forward)
            # Stage 2 时 INN 被冻结，此处相当于 Inference 模式，产生固定的 stego 分布
            stego_latent = model.embed(cover, secret)
            
            # 3. 模拟攻击
            if noise_std > 0:
                stego_damaged = simulate_vae_attack(stego_latent, noise_std=noise_std)
            else:
                stego_damaged = stego_latent # Stage 1 也就是无损
            
            # 4. 提取与矫正
            # Stage 1 时 Rectifier 被冻结(或初始化状态)，不起作用
            recovered_all, z_rectified = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # 5. Loss 计算
            loss_secret = torch.mean((rec_secret - secret) ** 2)
            loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
            loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
            
            # 按照阶段权重组合
            total_loss = (w_sec * loss_secret + 
                          w_stego * loss_stego + 
                          w_rect * loss_rect)
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            global_step += 1
            
            if global_step % 20 == 0:
                logger.info(f"Ep[{current_epoch}] {stage[:7]} | "
                            f"L_all:{total_loss:.4f} "
                            f"L_sec:{loss_secret:.4f} "
                            f"L_rect:{loss_rect:.4f}")

        # 保存阶段性模型
        if current_epoch in [stage1_end, stage2_end, cfg.EPOCHS]:
            save_name = f"model_{stage.split(' ')[2].replace('(', '').replace(')', '')}.pth"
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/{save_name}")
            logger.info(f"阶段完成，模型已保存: {save_name}")
        
        # 定期保存
        if current_epoch % 10 == 0:
            torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/epoch_{current_epoch}.pth")

if __name__ == "__main__":
    train()