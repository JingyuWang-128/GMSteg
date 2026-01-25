# config.py
import torch

class Config:
    # --- 基础配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    
    # --- 数据集路径 (请修改为您本地的真实路径) ---
    # 训练集: DIV2K (推荐使用 HR 高清图)
    DIV2K_TRAIN_PATH = "./data/DIV2K/train"
    # 验证集: DIV2K Valid
    DIV2K_VALID_PATH = "./data/DIV2K/valid"
    
    # 测试集 1: COCO (通常使用 val2017)
    COCO_TEST_PATH = "./data/test/coco"
    # 测试集 2: ImageNet (通常使用 val)
    IMAGENET_TEST_PATH = "./data/test/imagenet"
    
    # --- VAE 设置 ---
    # 推荐下载后改为本地路径: "./pretrained_models/sd-vae-ft-mse"
    VAE_ID = "stabilityai/sd-vae-ft-mse" 
    VAE_SCALE_FACTOR = 0.18215
    
    # --- 训练参数 ---
    IMAGE_SIZE = 256      # 训练时的裁剪尺寸
    BATCH_SIZE = 8        # DIV2K 图片很大，如果显存不够请调小
    NUM_WORKERS = 4
    LR = 1e-4
    EPOCHS = 100
    
    # --- 模型参数 ---
    LATENT_DIM = 4
    LATENT_SIZE = 32      # 256 // 8
    INN_CHANNELS = 8
    INN_BLOCKS = 4
    MAMBA_D_MODEL = 64
    
    # --- 结果保存 ---
    LOG_DIR = "./logs"
    CHECKPOINT_DIR = "./checkpoints"
    RESULT_DIR = "./results"
    
    # --- Loss 权重 ---
    LAMBDA_SECRET = 10.0
    LAMBDA_STEGO = 1.0
    LAMBDA_RECT = 5.0

cfg = Config()