import torch  # <--- 新增这行，修复 NameError
import torch.nn as nn
from .inn_block import LatentINNBlock
from .rectifier import DriftRectifier

class GenMambaINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # INN 主干: 堆叠多个可逆块
        self.inn_blocks = nn.ModuleList([
            LatentINNBlock(config.INN_CHANNELS, config.MAMBA_D_MODEL) 
            for _ in range(config.INN_BLOCKS)
        ])
        
        # 漂移矫正模块
        self.rectifier = DriftRectifier(config.INN_CHANNELS, config.MAMBA_D_MODEL)

    def forward(self, cover, secret):
        # 训练时的前向传播仅包含嵌入过程，
        # 提取过程和矫正过程在 Loss 计算或 Test 中单独调用
        return self.embed(cover, secret)

    def embed(self, cover, secret):
        # 简单拼接输入 (实际可改进为 Haar 变换融合)
        z = torch.cat([cover, secret], dim=1) 
        for block in self.inn_blocks:
            z = block(z, rev=False)
        return z

    def extract(self, stego_damaged):
        # 1. 先矫正 (Rectify)
        z_rectified = self.rectifier(stego_damaged)
        
        # 2. 后提取 (Invert)
        z = z_rectified
        for block in reversed(self.inn_blocks):
            z = block(z, rev=True)
            
        return z, z_rectified