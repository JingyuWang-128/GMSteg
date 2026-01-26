# utils/visualize.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def plot_loss_curves(log_file, save_path="results/loss_curve.png"):
    """
    读取日志文件并绘制训练损失曲线
    """
    if not os.path.exists(log_file):
        print(f"⚠️ 提示: 未找到日志文件 {log_file}，跳过 Loss 绘图。")
        return

    loss_all, loss_sec, loss_rect = [], [], []
    steps = []
    
    print(f"正在解析日志: {log_file} ...")
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析格式: "L_all:0.1234 L_sec:0.0123 L_rect:0.0012"
            if "L_all" in line:
                try:
                    all_val = float(re.search(r"L_all:([\d\.]+)", line).group(1))
                    sec_val = float(re.search(r"L_sec:([\d\.]+)", line).group(1))
                    rect_val = float(re.search(r"L_rect:([\d\.]+)", line).group(1))
                    
                    loss_all.append(all_val)
                    loss_sec.append(sec_val)
                    loss_rect.append(rect_val)
                    steps.append(len(steps))
                except Exception:
                    continue

    if len(steps) == 0:
        print("⚠️ 未能在日志中解析到 Loss 数据，请检查日志格式。")
        return

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(loss_all, label='Total Loss', alpha=0.6)
    plt.plot(loss_sec, label='Secret Recovery Loss', alpha=0.8, linewidth=1.5)
    plt.plot(loss_rect, label='Drift Rectification Loss', alpha=0.8, linewidth=1.5)
    
    plt.title("GenMamba-INN Training Loss Curves")
    plt.xlabel("Logging Steps")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"✅ 损失曲线已保存至: {save_path}")

def latents_to_rgb(vae, latents):
    """ 将潜码解码为 RGB 图片 (numpy uint8, 0-255) """
    # 1. 反缩放
    # latents = latents / 0.18215
    
    # 2. VAE 解码
    with torch.no_grad():
        imgs = vae.decode(latents).sample
    
    # 3. 后处理: [-1, 1] -> [0, 255]
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).astype(np.uint8)
    return imgs

def visualize_generation(model, vae, dataloader, device, save_path="results/visual_demo.png"):
    """
    随机抽取样本并可视化: Cover | Secret | Stego | Recovered | Residual
    """
    from train import images_to_latents # 避免循环引用

    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. 获取一个 Batch 数据
    try:
        images, _ = next(iter(dataloader))
    except StopIteration:
        print("数据加载失败")
        return

    if images.shape[0] < 4:
        print("Batch size 太小，无法展示两对图片，跳过可视化。")
        return

    # 取前两对图片 (Batch 中的前4张，0,1做Cover, 2,3做Secret)
    img_cover = images[0:2].to(device)
    img_secret = images[2:4].to(device)
    
    print(f"正在生成可视化演示图 (保存至 {save_path})...")
    
    with torch.no_grad():
        # --- 步骤 A: 编码 ---
        # 注意：这里直接用 train.py 里的函数比较方便
        z_cover = images_to_latents(vae, img_cover)
        z_secret = images_to_latents(vae, img_secret)
        
        # --- 步骤 B: 隐写 (Embed) ---
        z_stego = model.embed(z_cover, z_secret)
        
        # --- 步骤 C: 提取 (Extract) ---
        # 这里不加噪声，展示模型的“理想恢复能力”
        recovered_all, _ = model.extract(z_stego)
        _, z_rec_secret = recovered_all.chunk(2, dim=1)
        
        # --- 步骤 D: 解码回 RGB ---
        vis_cover = latents_to_rgb(vae, z_cover)      # 原始载体
        vis_secret = latents_to_rgb(vae, z_secret)    # 原始秘密
        vis_stego = latents_to_rgb(vae, z_stego)      # 隐写后的图
        vis_rec = latents_to_rgb(vae, z_rec_secret)   # 恢复的秘密
        
        # --- 步骤 E: 计算残差 (Stego - Cover) ---
        # 放大 10 倍以便肉眼观察隐写痕迹
        diff = np.abs(vis_stego.astype(int) - vis_cover.astype(int)) * 10
        vis_diff = np.clip(diff, 0, 255).astype(np.uint8)

    # 2. 拼图 (生成 2 行)
    # 每一行的顺序: [原始载体] [原始秘密] [加密载体] [恢复秘密] [残差(x10)]
    rows = []
    for i in range(2):
        # 水平拼接单行
        row = np.hstack([
            vis_cover[i], 
            vis_secret[i], 
            vis_stego[i], 
            vis_rec[i],
            vis_diff[i]
        ])
        rows.append(row)
    
    # 垂直拼接两行
    final_grid = np.vstack(rows)
    
    # 3. 保存为 BGR 格式 (OpenCV 标准)
    cv2.imwrite(save_path, cv2.cvtColor(final_grid, cv2.COLOR_RGB2BGR))
    
    print(f"✅ 可视化结果已保存! 列顺序: Cover | Secret | Stego | Recovered | Residual(x10)")