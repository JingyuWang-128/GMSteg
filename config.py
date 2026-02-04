# config.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch

class Config:
    # --- 基础配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    DIV2K_TRAIN_PATH = "./data/DIV2K/train"
    DIV2K_VALID_PATH = "./data/DIV2K/valid"
    COCO_TEST_PATH = "./data/coco"
    IMAGENET_TEST_PATH = "./data/imagenet"
    
    # VAE 路径
    VAE_ID = "./pretrained_models/sd-vae-ft-mse"
    VAE_SCALE_FACTOR = 0.18215
    
    # --- 分阶段训练设置 ---
    EPOCHS_STAGE1 = 150  # Warm-up INN (Focus on embedding)
    EPOCHS_STAGE2 = 150  # Train Rectifier (Focus on robustness)
    EPOCHS_STAGE3 = 200  # Joint Fine-tuning
    EPOCHS = EPOCHS_STAGE1 + EPOCHS_STAGE2 + EPOCHS_STAGE3
    
    # --- 训练参数 ---
    IMAGE_SIZE = 256
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LR = 1e-4
    
    # --- 模型参数 ---
    LATENT_DIM = 4
    INN_CHANNELS = 8
    INN_BLOCKS = 4
    MAMBA_D_MODEL = 64
    
    # [新增] 矫正器迭代步数 (Iterative Rectification Steps)
    RECT_STEPS = 3
    
    # --- Loss 权重 ---
    # 提高 Secret 权重以防止模型偷懒
    LAMBDA_SECRET = 100.0 
    LAMBDA_STEGO = 0.1
    LAMBDA_RECT = 5.0
    
    # --- 结果保存 ---
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    RESULT_DIR = "./results"

    def __init__(self):
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULT_DIR, exist_ok=True)

cfg = Config()