# config.py
import torch

class Config:
    # --- 硬件与环境 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # --- 路径设置 ---
    # 请修改为你存放真实图片的路径 (例如 COCO, DIV2K, ImageNet)
    # 结构要求: ./datasets/coco/train/xxx.jpg
    DATASET_PATH = "./datasets/coco" 
    # 预训练 VAE 模型 ID (来自 HuggingFace)
    VAE_ID = "stabilityai/sd-vae-ft-mse"
    
    # --- 结果保存 ---
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    
    # --- 潜空间参数 (Stable Diffusion 标准) ---
    LATENT_DIM = 4        # SD VAE 通道数
    IMAGE_SIZE = 256      # 输入图片大小 (会压缩为 32x32 的 Latent)
    LATENT_SIZE = 32      # 256 / 8 = 32
    VAE_SCALE_FACTOR = 0.18215 # SD 标准缩放因子
    
    # --- GenMamba 模型参数 ---
    INN_CHANNELS = 8      # Cover(4) + Secret(4)
    INN_BLOCKS = 4        # 可逆块深度
    MAMBA_D_MODEL = 64    # Mamba 隐藏层维度
    
    # --- 训练超参数 ---
    LR = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 8        # 显存允许可调大
    NUM_WORKERS = 4
    
    # --- 损失权重 (根据实验经验调整) ---
    LAMBDA_SECRET = 10.0  # 秘密信息恢复最重要
    LAMBDA_STEGO = 1.0    # 载体保真度
    LAMBDA_RECT = 5.0     # 漂移矫正 (创新点3核心)

cfg = Config()