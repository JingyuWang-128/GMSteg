# test.py (更新版)
import torch
import numpy as np
from config import cfg
from models import GenMambaINN
from utils.datasets import get_dataloader
from train import load_vae, images_to_latents

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
    # 1. 加载模型
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    
    # 加载最新权重 (假设已训练完)
    ckpt_path = f"{cfg.CHECKPOINT_DIR}/epoch_{cfg.EPOCHS}.pth"
    # ckpt_path = f"{cfg.CHECKPOINT_DIR}/gen_mamba_latest.pth" # 或者这个
    try:
        model.load_state_dict(torch.load(ckpt_path))
        print(f"已加载权重: {ckpt_path}")
    except:
        print("未找到权重，请先运行 train.py！")
        return
    
    model.eval()
    
    # 2. 测试 COCO
    coco_loader = get_dataloader(cfg.COCO_TEST_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=False)
    evaluate_on_dataset(model, vae, coco_loader, "COCO 2017", device)
    
    # 3. 测试 ImageNet
    imagenet_loader = get_dataloader(cfg.IMAGENET_TEST_PATH, cfg.IMAGE_SIZE, cfg.BATCH_SIZE, is_train=False)
    evaluate_on_dataset(model, vae, imagenet_loader, "ImageNet Val", device)

if __name__ == "__main__":
    run_full_test()