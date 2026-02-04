import torch
import torch.nn as nn
from utils.frequency import analyze_frequency_energy
# 使用新的双门控 Mamba
from .mamba_block import SemanticTextureGatedMamba 

class LatentINNBlock(nn.Module):
    def __init__(self, in_channels, d_model=64):
        super().__init__()
        self.split_len = in_channels // 2
        
        # 使用 SemanticTextureGatedMamba
        self.mamba_s = SemanticTextureGatedMamba(self.split_len, d_model)
        self.mamba_t = SemanticTextureGatedMamba(self.split_len, d_model)

    def forward(self, x, rev=False):
        B, C, H, W = x.shape
        x1, x2 = x.chunk(2, dim=1)
        
        freq_map = analyze_frequency_energy(x1)
        x1_flat = x1.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        
        if not rev: # Embed
            s = self.mamba_s(x1_flat, freq_map)
            t = self.mamba_t(x1_flat, freq_map)
            
            s = s.view(B, H, W, -1).permute(0, 3, 1, 2)
            t = t.view(B, H, W, -1).permute(0, 3, 1, 2)
            
            # 🔥 必须加 Tanh 锁，防止梯度爆炸 🔥
            s = 2.0 * torch.tanh(s)
            
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            return torch.cat([y1, y2], dim=1)
            
        else: # Extract
            s = self.mamba_s(x1_flat, freq_map)
            t = self.mamba_t(x1_flat, freq_map)
            
            s = s.view(B, H, W, -1).permute(0, 3, 1, 2)
            t = t.view(B, H, W, -1).permute(0, 3, 1, 2)
            
            s = 2.0 * torch.tanh(s)
            
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([y1, y2], dim=1)