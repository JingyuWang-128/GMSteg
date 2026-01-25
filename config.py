import torch

class Config:
    # --- 基础设置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # --- 潜空间设置 (Stable Diffusion VAE) ---
    # SD 的 Latent 通常是 4 通道，尺寸根据输入图像缩小 8 倍 (例如 512x512 -> 64x64)
    LATENT_DIM = 4  
    LATENT_SIZE = 64
    
    # --- 模型架构参数 ---
    # INN 的总通道数 = Cover(4) + Secret(4) = 8
    INN_CHANNELS = 8
    INN_BLOCKS = 4        # 可逆块的数量 (Depth)
    MAMBA_D_MODEL = 64    # Mamba 内部隐藏层维度
    
    # --- 训练超参数 ---
    LR = 2e-4
    EPOCHS = 100
    BATCH_SIZE = 8
    
    # --- 损失权重 ---
    LAMBDA_SECRET = 10.0  # 秘密信息恢复损失权重
    LAMBDA_STEGO = 1.0    # 隐写潜码保真度权重
    LAMBDA_RECT = 5.0     # 漂移矫正权重 (创新点3)

    # --- 路径 ---
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"

cfg = Config()