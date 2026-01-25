import torch
import torch.nn.functional as F

def analyze_frequency_energy(latent_map):
    """
    计算潜空间特征图的局部高频能量，用于生成频率门控。
    Input:  [B, C, H, W]
    Output: [B, 1, H, W] (归一化能量图)
    """
    # 使用一阶差分近似高频梯度 (类似 Sobel 算子)
    # 水平梯度
    grad_h = torch.abs(latent_map[:, :, :, :-1] - latent_map[:, :, :, 1:])
    grad_h = F.pad(grad_h, (0, 1, 0, 0)) # 补齐
    
    # 垂直梯度
    grad_v = torch.abs(latent_map[:, :, :-1, :] - latent_map[:, :, 1:, :])
    grad_v = F.pad(grad_v, (0, 0, 0, 1))
    
    # 综合能量 (L1 Norm)
    energy = grad_h + grad_v
    
    # 跨通道平均，得到空间上的能量分布
    energy = energy.mean(dim=1, keepdim=True)
    
    # 归一化到 0~1 (Min-Max Scaling)
    # 加上 eps 防止除零
    B = energy.shape[0]
    energy_flat = energy.view(B, -1)
    min_val = energy_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_val = energy_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    norm_energy = (energy - min_val) / (max_val - min_val + 1e-6)
    
    return norm_energy