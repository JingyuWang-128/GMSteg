import torch
import torch.nn as nn
from .inn_block import LatentINNBlock
from .rectifier import DriftRectifier

class GenMambaINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        # INN 主干
        self.inn_blocks = nn.ModuleList([
            LatentINNBlock(config.INN_CHANNELS, config.MAMBA_D_MODEL) 
            for _ in range(config.INN_BLOCKS)
        ])
        
        # 漂移矫正模块
        self.rectifier = DriftRectifier(config.INN_CHANNELS, config.MAMBA_D_MODEL)

    def forward(self, cover, secret):
        return self.embed(cover, secret)

    def embed(self, cover, secret):
        z = torch.cat([cover, secret], dim=1) 
        
        for i, block in enumerate(self.inn_blocks):
            # 1. 可逆变换
            z = block(z, rev=False)
            
            # 2. 【新增】通道交换 (Swap)
            # 必须交换，否则 Cover 永远不会被修改！
            z = torch.cat([z.chunk(2, 1)[1], z.chunk(2, 1)[0]], dim=1)
            
        return z

    def extract(self, stego_damaged):
        # 1. 先矫正
        z_rectified = self.rectifier(stego_damaged)
        
        z = z_rectified
        # 2. 后提取 (逆序)
        for i, block in enumerate(reversed(self.inn_blocks)):
            
            # 2.1 【新增】撤销通道交换 (Undo Swap)
            z = torch.cat([z.chunk(2, 1)[1], z.chunk(2, 1)[0]], dim=1)
            
            # 2.2 撤销可逆变换
            z = block(z, rev=True)
            
        return z, z_rectified