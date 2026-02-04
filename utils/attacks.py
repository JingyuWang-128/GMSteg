import torch
import torch.nn.functional as F
import numpy as np

def attack_in_rgb(image_tensor, attack_type='random'):
    """
    在 RGB 像素域进行攻击模拟
    输入: image_tensor [B, 3, H, W], 范围 [-1, 1]
    输出: attacked_tensor [B, 3, H, W], 范围 [-1, 1]
    """
    # 1. 转为 [0, 1] 方便处理
    img = (image_tensor + 1) / 2.0
    
    # 2. 选择攻击模式
    if attack_type == 'random':
        # 概率分布: 噪声(30%), 裁切(20%), 遮挡(20%), 模糊(20%), JPEG模拟(10%)
        attack_type = np.random.choice(
            ['noise', 'crop', 'dropout', 'blur', 'jpeg'], 
            p=[0.3, 0.2, 0.2, 0.2, 0.1]
        )
    
    # --- 攻击实现 ---
    if attack_type == 'noise':
        # 高斯噪声
        noise = torch.randn_like(img) * 0.05
        img = torch.clamp(img + noise, 0, 1)
        
    elif attack_type == 'crop':
        # 随机边缘裁切 (模拟截图不全)
        B, C, H, W = img.shape
        ratio = np.random.uniform(0.1, 0.25) # 裁掉 10%-25%
        h_cut = int(H * ratio)
        w_cut = int(W * ratio)
        mask = torch.zeros_like(img)
        # 保留中心区域
        mask[:, :, h_cut:H-h_cut, w_cut:W-w_cut] = 1.0
        img = img * mask
        
    elif attack_type == 'dropout':
        # 随机遮挡块 (Coarse Dropout)
        B, C, H, W = img.shape
        for i in range(B):
            # 随机遮挡 32x32 的块
            h_start = np.random.randint(0, H-32)
            w_start = np.random.randint(0, W-32)
            img[i, :, h_start:h_start+32, w_start:w_start+32] = 0.0
            
    elif attack_type == 'blur':
        # 平均模糊
        img = F.avg_pool2d(img, kernel_size=3, stride=1, padding=1)
        
    elif attack_type == 'jpeg':
        # 模拟量化噪声 (Rounding)
        steps = 32.0 
        img = torch.round(img * steps) / steps

    # 3. 转回 [-1, 1]
    return torch.clamp(img * 2.0 - 1.0, -1, 1)

# 保留旧的 latent 攻击函数以防兼容性问题，但主要用上面的
def simulate_vae_attack(latent, noise_std=0.1):
    return latent + torch.randn_like(latent) * noise_std