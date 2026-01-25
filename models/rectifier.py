import torch
import torch.nn as nn
from .mamba_block import SimplifiedMambaKernel 

class DriftRectifier(nn.Module):
    """
    [Innovation 3] Mamba 引导的潜码漂移矫正
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        # 使用深层 Mamba 进行去噪/修复
        self.embedding = nn.Linear(channels, d_model)
        self.body = nn.Sequential(
            SimplifiedMambaKernel(d_model),
            SimplifiedMambaKernel(d_model) # 堆叠两层增加感受野
        )
        self.head = nn.Linear(d_model, channels)

    def forward(self, z_damaged):
        """
        输入: 受损潜码 [B, C, H, W]
        输出: 矫正后潜码 [B, C, H, W]
        """
        B, C, H, W = z_damaged.shape
        z_flat = z_damaged.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # 预测残差 (Drift)
        feat = self.embedding(z_flat)
        feat = self.body(feat)
        drift = self.head(feat)
        
        drift = drift.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # 减去预测的漂移量
        return z_damaged - drift