import torch
import torch.optim as optim
from config import cfg
from models import GenMambaINN  
from utils.logger import get_logger
from utils.attacks import simulate_vae_attack

def train():
    logger = get_logger("GenMamba-Train", cfg.LOG_DIR)
    logger.info("初始化 GenMamba-INN 模型...")
    
    model = GenMambaINN(cfg).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LR)
    
    logger.info("开始模拟训练...")
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        
        # 模拟数据 (Batch, C, H, W)
        cover = torch.randn(cfg.BATCH_SIZE, cfg.LATENT_DIM, cfg.LATENT_SIZE, cfg.LATENT_SIZE).to(cfg.DEVICE)
        secret = torch.randn(cfg.BATCH_SIZE, cfg.LATENT_DIM, cfg.LATENT_SIZE, cfg.LATENT_SIZE).to(cfg.DEVICE)
        
        # 1. 嵌入 (Embedding)
        stego_latent = model.embed(cover, secret)
        
        # 2. 模拟攻击 (Attack Simulation) - 训练 Rectifier 必须步骤
        stego_damaged = simulate_vae_attack(stego_latent, noise_std=0.1)
        
        # 3. 提取与矫正 (Extract & Rectify)
        recovered_all, z_rectified = model.extract(stego_damaged)
        rec_cover, rec_secret = recovered_all.chunk(2, dim=1)
        
        # --- Loss 计算 ---
        # L_sec: 秘密信息恢复损失
        loss_secret = torch.mean((rec_secret - secret) ** 2)
        # L_stego: 隐写不可感知损失 (约束潜码不偏离太远)
        loss_stego = torch.mean((stego_latent[:, :4] - cover) ** 2)
        # L_rect: 漂移矫正损失 (监督 LDR 模块去噪)
        loss_rect = torch.mean((z_rectified - stego_latent) ** 2)
        
        total_loss = (cfg.LAMBDA_SECRET * loss_secret + 
                      cfg.LAMBDA_STEGO * loss_stego + 
                      cfg.LAMBDA_RECT * loss_rect)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{cfg.EPOCHS}] "
                        f"Loss: {total_loss.item():.6f} | "
                        f"Sec: {loss_secret.item():.6f} | "
                        f"Rect: {loss_rect.item():.6f}")

    # 保存模型
    torch.save(model.state_dict(), f"{cfg.CHECKPOINT_DIR}/gen_mamba_latest.pth")
    logger.info("训练完成，模型已保存。")

if __name__ == "__main__":
    train()