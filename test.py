# test.py
import torch
import cv2
import numpy as np
from diffusers import AutoencoderKL
from config import cfg
from models import GenMambaINN
from train import load_vae, images_to_latents

def latents_to_images(vae, latents):
    """ 将潜码还原为图片用于保存 """
    latents = latents / cfg.VAE_SCALE_FACTOR
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    return (image * 255).astype(np.uint8)

def run_test():
    device = cfg.DEVICE
    
    # 1. 加载模型
    vae = load_vae(device)
    model = GenMambaINN(cfg).to(device)
    
    # 尝试加载权重
    try:
        model.load_state_dict(torch.load(f"{cfg.CHECKPOINT_DIR}/epoch_100.pth"))
        print("成功加载训练权重。")
    except:
        print("未找到权重，使用随机初始化进行演示...")
        
    model.eval()
    
    # 2. 构造测试数据 (这里用随机噪声模拟图片，实际请读取本地图片)
    # 模拟两张 256x256 的图片
    cover_img = torch.randn(1, 3, 256, 256).to(device) 
    secret_img = torch.randn(1, 3, 256, 256).to(device)
    
    print("正在进行潜空间投影...")
    cover = images_to_latents(vae, cover_img)
    secret = images_to_latents(vae, secret_img)
    
    # 3. 执行隐写流程
    print("正在执行 Frequency-Gated 嵌入...")
    stego_latent = model.embed(cover, secret)
    
    print("正在执行 漂移矫正与提取...")
    # 模拟一点噪声
    stego_damaged = stego_latent + torch.randn_like(stego_latent) * 0.05
    recovered_all, z_rectified = model.extract(stego_damaged)
    _, rec_secret = recovered_all.chunk(2, dim=1)
    
    # 4. 还原为图片
    print("正在解码回像素空间...")
    img_stego = latents_to_images(vae, stego_latent)[0]
    img_secret_rec = latents_to_images(vae, rec_secret)[0]
    
    # 保存结果
    cv2.imwrite("result_stego.png", cv2.cvtColor(img_stego, cv2.COLOR_RGB2BGR))
    cv2.imwrite("result_secret_rec.png", cv2.cvtColor(img_secret_rec, cv2.COLOR_RGB2BGR))
    print("演示完成！结果已保存为 result_stego.png 和 result_secret_rec.png")

if __name__ == "__main__":
    run_test()