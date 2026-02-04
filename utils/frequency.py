import torch
import torch.nn.functional as F

def analyze_frequency_energy(latent_map, edge_suppression_strength=1.0):
    """
    计算潜空间特征图的局部纹理能量，并显式抑制强边缘。
    
    Input:  latent_map [B, C, H, W]
    Output: freq_map   [B, 1, H, W] (归一化且抑制边缘后的能量图)
    """
    # 1. 计算一阶差分近似梯度 (Sobel-like)
    # -------------------------------------------------------------------------
    # 水平梯度: |x[i+1] - x[i]|
    grad_h = torch.abs(latent_map[:, :, :, :-1] - latent_map[:, :, :, 1:])
    grad_h = F.pad(grad_h, (0, 1, 0, 0)) # 补齐边界
    
    # 垂直梯度: |y[j+1] - y[j]|
    grad_v = torch.abs(latent_map[:, :, :-1, :] - latent_map[:, :, 1:, :])
    grad_v = F.pad(grad_v, (0, 0, 0, 1))
    
    # 综合能量 (L1 Norm)
    energy = grad_h + grad_v
    
    # 跨通道平均，得到空间上的能量分布 [B, 1, H, W]
    energy = energy.mean(dim=1, keepdim=True)
    
    # 2. 归一化到 0~1 (Min-Max Scaling)
    # -------------------------------------------------------------------------
    B = energy.shape[0]
    energy_flat = energy.view(B, -1)
    min_val = energy_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_val = energy_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    # 加上 eps 防止除零
    norm_energy = (energy - min_val) / (max_val - min_val + 1e-6)
    
    # 3. 【核心创新】显式边缘抑制 (Explicit Edge Suppression)
    # -------------------------------------------------------------------------
    # 逻辑：
    # - 强边缘 (Strong Edges): norm_energy 接近 1.0 (梯度最大)
    # - 平滑区 (Smooth):       norm_energy 接近 0.0 (梯度最小)
    # - 纹理区 (Texture):      norm_energy 位于 0.3~0.7 之间
    #
    # 我们使用抛物线函数 y = 4 * x * (1 - x) 实现带通滤波：
    # 当 x=0 (平滑) -> y=0
    # 当 x=1 (边缘) -> y=0 (被抑制!)
    # 当 x=0.5(纹理)-> y=1 (最大化)
    
    # 引入指数 alpha 调节抑制强度 (edge_suppression_strength)
    # alpha 越大，对边缘的惩罚越重，只保留最中间的细腻纹理
    
    suppressed_energy = 4.0 * norm_energy * (1.0 - norm_energy)
    
    # 加上次幂以进一步锐化选择范围 (可选)
    if edge_suppression_strength != 1.0:
        suppressed_energy = torch.pow(suppressed_energy, edge_suppression_strength)
        
    return suppressed_energy