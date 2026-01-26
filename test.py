# test.py (最终完善版)
import torch
import numpy as np
import os
from config import cfg
from models import GenMambaINN
from utils.datasets import get_dataloader
from train import load_vae, images_to_latents
# 引入新写的可视化工具
from utils.visualize import plot_loss_curves, visualize_generation

def evaluate_on_dataset(model, vae, dataloader, dataset_name, device):
    print(f"\n>>> 正在测试数据集: {dataset_name} ...")
    mse_secret_list = []
    mse_stego_list = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            if images.shape[0] < 2: continue
            
            split = images.shape[0] // 2
            cover = images_to_latents(vae, images[:split])
            secret = images_to_latents(vae, images[split:split*2])
            
            # 1. 嵌入
            stego = model.embed(cover, secret)
            
            # 2. 模拟攻击 (测试鲁棒性)
            stego_damaged = stego + torch.randn_like(stego) * 0.1 # VAE 模拟噪声
            
            # 3. 提取
            recovered_all, _ = model.extract(stego_damaged)
            _, rec_secret = recovered_all.chunk(2, dim=1)
            
            # 记录指标 (Latent MSE)
            mse_sec = torch.mean((rec_secret - secret) ** 2).item()
            mse_stego = torch.mean((stego[:, :4] - cover) ** 2).item()
            
            mse_secret_list.append(mse_sec)
            mse_stego_list.append(mse_stego)
            
            if i % 10 == 0:
                print(f"Batch {i}: Secret MSE = {mse_sec:.6f}")
                
    avg_sec = np.mean(mse_secret_list)
    avg_stego = np.mean(mse_stego_list)
    print(f"=== {dataset_name} 测试结果 ===")
    print(f"平均秘密信息 MSE: {avg_sec:.6f}")
    print(f"平均载体隐写 MSE: {avg_stego:.6f}")
    return avg_sec

def run_full_test():
    device = cfg.DEVICE
    # 确保结果目录存在
    if not hasattr(cfg, 'RESULT_DIR'):
        cfg.RESULT_DIR = "./results"
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    # 1. 加载模型
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    
    # 加载最新权重
    # 优先加载指定 Epoch，如果不存在则加载 latest
    ckpt_path = f"{cfg.CHECKPOINT_DIR}/epoch_{cfg.EPOCHS}.pth"
    if not os.path.exists(ckpt_path):
        ckpt_path = f"{cfg.CHECKPOINT_DIR}/gen_mamba_latest.pth"
        
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"已加载权重: {ckpt_path}")
    except:
        print(f"❌ 未找到权重文件 {ckpt_path}，请先运行 train.py！")
        return
    
    model.eval()
    
    # --- Part A: 定量测试 (Quantitative) ---
    # 2. 测试 COCO
    if hasattr(cfg, 'COCO_TEST_PATH') and os.path.exists(cfg.COCO_TEST_PATH):
        coco_loader = get_dataloader(cfg.COCO_TEST_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=False)
        evaluate_on_dataset(model, vae, coco_loader, "COCO 2017", device)
    else:
        print("⚠️ 未找到 COCO 数据集路径，跳过 COCO 测试。")
    
    # 3. 测试 ImageNet
    if hasattr(cfg, 'IMAGENET_TEST_PATH') and os.path.exists(cfg.IMAGENET_TEST_PATH):
        imagenet_loader = get_dataloader(cfg.IMAGENET_TEST_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=False)
        evaluate_on_dataset(model, vae, imagenet_loader, "ImageNet Val", device)
    else:
        print("⚠️ 未找到 ImageNet 数据集路径，跳过 ImageNet 测试。")

    # --- Part B: 可视化 (Visualization) ---
    print("\n>>> 开始生成可视化报告...")
    
    # 1. 绘制 Loss 曲线
    log_file = os.path.join(cfg.LOG_DIR, "train.log")
    plot_loss_curves(log_file, save_path=os.path.join(cfg.RESULT_DIR, "loss_curve.png"))
    
    # 2. 生成图像可视化 (使用 COCO 或 ImageNet 的数据)
    # 如果都没有，尝试用 DIV2K Valid，如果还没有就报错
    viz_loader = None
    if hasattr(cfg, 'COCO_TEST_PATH') and os.path.exists(cfg.COCO_TEST_PATH):
        viz_loader = get_dataloader(cfg.COCO_TEST_PATH, cfg.IMAGE_SIZE, batch_size=4, is_train=False)
    elif hasattr(cfg, 'DIV2K_VALID_PATH') and os.path.exists(cfg.DIV2K_VALID_PATH):
        viz_loader = get_dataloader(cfg.DIV2K_VALID_PATH, cfg.IMAGE_SIZE, batch_size=4, is_train=False)
        
    if viz_loader:
        visualize_generation(model, vae, viz_loader, device, 
                             save_path=os.path.join(cfg.RESULT_DIR, "visual_demo.png"))
    else:
        print("⚠️ 无可用数据集用于可视化，跳过图片生成。")

if __name__ == "__main__":
    run_full_test()