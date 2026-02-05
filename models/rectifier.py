# models/rectifier.py
import torch
import torch.nn as nn
# 导入修改后的通用 Mamba 层构建函数
from .mamba_block import Mamba

class DriftRectifier(nn.Module):
    """
    [创新点 3] Mamba 引导的潜空间漂移矫正 (LDR)
    利用 Mamba 的全局上下文能力，预测并修复由 VAE 重构或攻击导致的特征漂移。
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        self.embedding = nn.Linear(channels, d_model)
        
        # 使用两层 Mamba 堆叠，增大感受野以捕捉全局漂移模式
        self.body = nn.Sequential(
            Mamba(d_model),
            nn.LayerNorm(d_model), # 增加 Norm 层提升稳定性
            Mamba(d_model),
            nn.LayerNorm(d_model)
        )
        
        self.head = nn.Linear(d_model, channels)

    def forward(self, z_damaged):
        """
        输入: 受损潜码 [B, C, H, W]
        输出: 矫正后潜码 [B, C, H, W]
        """
        B, C, H, W = z_damaged.shape
        # 序列化: [B, C, H, W] -> [B, H*W, C] (注意 Mamba 需要 (B, L, D))
        z_flat = z_damaged.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        # 预测漂移量 (Residual Learning)
        feat = self.embedding(z_flat)
        feat = self.body(feat)
        drift = self.head(feat)
        
        # 恢复空间维度
        drift = drift.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # 核心逻辑: 矫正 = 输入 - 预测的漂移
        return z_damaged - drift