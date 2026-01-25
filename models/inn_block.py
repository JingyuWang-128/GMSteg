import torch
import torch.nn as nn
from utils.frequency import analyze_frequency_energy
from .mamba_block import FrequencyGatedSSM

class LatentINNBlock(nn.Module):
    """
    [Innovation 1] 潜空间可逆耦合层
    """
    def __init__(self, in_channels, d_model=64):
        super().__init__()
        self.split_len = in_channels // 2
        
        # 变换函数 s() 和 t() 使用 FrequencyGatedSSM
        self.mamba_s = FrequencyGatedSSM(self.split_len, d_model)
        self.mamba_t = FrequencyGatedSSM(self.split_len, d_model)

    def forward(self, x, rev=False):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x1, x2 = x.chunk(2, dim=1)
        
        # 准备频率上下文
        # 假设 x1 是控制通道，利用 x1 的纹理决定 x2 的嵌入强度
        freq_map = analyze_frequency_energy(x1)
        
        # 序列化供 Mamba 使用
        x1_flat = x1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        if not rev: # 前向嵌入
            s = self.mamba_s(x1_flat, freq_map)
            t = self.mamba_t(x1_flat, freq_map)
            
            # Reshape 回空间维度
            s = s.view(B, H, W, -1).permute(0, 3, 1, 2)
            t = t.view(B, H, W, -1).permute(0, 3, 1, 2)
            
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            return torch.cat([y1, y2], dim=1)
            
        else: # 逆向提取
            s = self.mamba_s(x1_flat, freq_map)
            t = self.mamba_t(x1_flat, freq_map)
            
            s = s.view(B, H, W, -1).permute(0, 3, 1, 2)
            t = t.view(B, H, W, -1).permute(0, 3, 1, 2)
            
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([y1, y2], dim=1)