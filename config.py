# config.py
import torch
import os

class Config:
    # --- 基础配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # --- 路径设置 (请确保路径正确) ---
    # 训练集: DIV2K
    DIV2K_TRAIN_PATH = "./data/DIV2K/train"
    # 验证集: DIV2K Valid
    DIV2K_VALID_PATH = "./data/DIV2K/valid"
    
    # [新增] 测试集路径 (用于 test.py)
    # 请根据您本地数据集实际位置修改
    COCO_TEST_PATH = "./data/coco"
    IMAGENET_TEST_PATH = "./data/imagenet"
    
    # 预训练 VAE 路径
    VAE_ID = "./pretrained_models/sd-vae-ft-mse" # 建议使用本地路径
    VAE_SCALE_FACTOR = 0.18215
    
    # --- 分阶段训练设置 (Curriculum Learning) ---
    # 总 Epochs = STAGE1 + STAGE2 + STAGE3
    # 建议设置: 150 -> 150 -> 200 = 500
    EPOCHS_STAGE1 = 150  # 阶段一: 仅训练 INN (无噪声)
    EPOCHS_STAGE2 = 150  # 阶段二: 仅训练 Rectifier (强噪声)
    EPOCHS_STAGE3 = 200  # 阶段三: 联合微调 (标准噪声)
    
    # 计算总 Epochs 用于 DataLoader
    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3
    
    # --- 训练参数 ---
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LR = 1e-4
    
    # --- 模型参数 ---
    LATENT_DIM = 4
    LATENT_SIZE = 32
    INN_CHANNELS = 8
    INN_BLOCKS = 4
    MAMBA_D_MODEL = 64
    
    # --- Loss 权重 (基准值) ---
    LAMBDA_SECRET = 10.0
    LAMBDA_STEGO = 1.0
    LAMBDA_RECT = 5.0
    
    # --- 结果保存 ---
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    # [新增] 可视化结果保存路径
    RESULT_DIR = "./results"

    def __init__(self):
        # 自动创建必要目录
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

cfg = Config()