import torch
from config import cfg
from models import GenMambaINN  
from utils.logger import get_logger
from utils.attacks import simulate_vae_attack

def test_demo():
    logger = get_logger("GenMamba-Test")
    logger.info("--- 运行 GenMamba-INN 完整流程演示 ---")
    
    # 初始化模型
    model = GenMambaINN(cfg).to(cfg.DEVICE)
    # 加载权重 (如果是实际运行，请取消注释)
    # model.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/gen_mamba_latest.pth"))
    model.eval()
    
    with torch.no_grad():
        # 1. 准备数据
        logger.info("[Step 1] 准备潜空间数据 (Cover & Secret Latents)")
        cover = torch.randn(1, 4, 64, 64).to(cfg.DEVICE)
        secret = torch.randn(1, 4, 64, 64).to(cfg.DEVICE)
        
        # 2. 嵌入 (Innovation 1 & 2)
        # 内部会调用 FrequencyGatedSSM 进行动态嵌入
        stego = model.embed(cover, secret)
        logger.info(f"[Step 2] 嵌入完成. Stego Latent Shape: {stego.shape}")
        
        # 3. 模拟强攻击 (Innovation 3 验证)
        # 模拟较大的 VAE 重构损耗
        stego_noisy = simulate_vae_attack(stego, noise_std=0.2)
        mse_attack = ((stego_noisy - stego)**2).mean().item()
        logger.info(f"[Step 3] 模拟 VAE/信道攻击完成. 引入误差 MSE: {mse_attack:.6f}")
        
        # 4. 提取与矫正
        recovered, z_rectified = model.extract(stego_noisy)
        rec_cover, rec_secret = recovered.chunk(2, dim=1)
        
        # 5. 结果评估
        mse_rectified = ((z_rectified - stego)**2).mean().item()
        mse_secret = ((rec_secret - secret)**2).mean().item()
        
        logger.info("--- 最终结果 ---")
        logger.info(f"漂移矫正模块 (LDR) 效果: 误差从 {mse_attack:.6f} 降至 {mse_rectified:.6f}")
        logger.info(f"秘密信息恢复 MSE: {mse_secret:.6f}")
        
        if mse_secret < mse_attack:
            logger.info("✅ 测试通过: 系统成功抵抗了攻击并恢复了信息。")
        else:
            logger.info("⚠️ 提示: 模型未充分训练，仅展示流程逻辑。")

if __name__ == "__main__":
    test_demo()