import torch
import random

def simulate_vae_attack(latent, vae=None, noise_std=0.1):
    """
    模拟真实的 VAE 回环攻击 (Latent -> Image -> Attack -> Latent)
    
    Args:
        latent: [B, C, H, W] 潜码
        vae: AutoencoderKL 模型实例 (如果为 None，则降级为简单的潜空间加噪)
        noise_std: 攻击强度 (RGB 域或 Latent 域的标准差)
    """
    # 模式 A: 快速模拟 (仅潜空间加噪)
    if vae is None:
        noise = torch.randn_like(latent) * noise_std
        return latent + noise
    
    # 模式 B: 真实回环攻击 (Real VAE Round-trip)
    # 这能捕捉到 VAE 重构带来的结构性漂移，完美契合 LDR 创新点
    
    # 1. 解码 (Decode): Latent -> Image (RGB)
    with torch.no_grad():
        # 注意：根据您的 train.py，latents 未缩放，直接解码即可
        # VAE decode 输出范围通常是 [-1, 1]
        decoded_images = vae.decode(latent).sample
        
    # 2. 图像域攻击 (Image Domain Attack)
    # 模拟信道噪声 (Gaussian Noise on Pixels)
    if noise_std > 0:
        noise_rgb = torch.randn_like(decoded_images) * noise_std
        attacked_images = decoded_images + noise_rgb
        # 必须截断回有效像素范围 [-1, 1]，模拟真实的图像存储限制
        attacked_images = torch.clamp(attacked_images, -1.0, 1.0)
    else:
        attacked_images = decoded_images

    # 3. 再编码 (Re-encode): Image (Attacked) -> Latent'
    with torch.no_grad():
        dist = vae.encode(attacked_images).latent_dist
        damaged_latent = dist.sample()
        
    return damaged_latent

def simulate_dropout_attack(latent, vae=None, drop_prob=0.3):
    """
    模拟真实的图像遮挡/裁剪攻击 (Latent -> Image -> Occlusion -> Latent)
    
    Args:
        latent: [B, C, H, W] 潜码
        vae: AutoencoderKL 模型
        drop_prob: 遮挡面积占总面积的比例 (近似值)
    """
    # 模式 A: 仅潜空间随机丢弃 (Fallback, 模拟信道丢包)
    if vae is None:
        mask = torch.bernoulli(torch.ones_like(latent) * (1 - drop_prob))
        return latent * mask

    # 模式 B: 真实图像域遮挡 (Real Image Occlusion)
    # 1. 解码 (Decode)
    with torch.no_grad():
        images = vae.decode(latent).sample
        # images range: [-1, 1] (通常)

    # 2. 图像域遮挡 (Apply Occlusion/Cutout on RGB Images)
    B, C, H, W = images.shape
    device = images.device
    
    # 创建一个遮挡掩码 (1=保留, 0=遮挡)
    mask = torch.ones((B, 1, H, W), device=device)
    
    for i in range(B):
        # 随机生成遮挡块的大小
        # 假设遮挡一个矩形区域，面积约为 drop_prob
        cut_w = int(W * (drop_prob ** 0.5)) 
        cut_h = int(H * (drop_prob ** 0.5))
        
        # 随机位置
        if cut_w < W and cut_h < H:
            x = random.randint(0, W - cut_w)
            y = random.randint(0, H - cut_h)
            
            # 将该区域置为 0 (黑色遮挡)
            mask[i, :, y:y+cut_h, x:x+cut_w] = 0.0
            
    # 应用遮挡 (模拟黑色块遮挡，如果是裁剪可以将背景设为其他颜色)
    # 这里用 -1 (黑色/深色, 取决于归一化) 填充，或者直接乘 0 变灰
    # 为了模拟信息彻底丢失，我们让像素值变成 -1 (接近纯黑) 或者 保持原值 * 0
    attacked_images = images * mask + (1 - mask) * -1.0 

    # 3. 再编码 (Re-encode)
    with torch.no_grad():
        dist = vae.encode(attacked_images).latent_dist
        damaged_latent = dist.sample()
        
    return damaged_latent