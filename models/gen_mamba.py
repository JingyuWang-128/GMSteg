import torch
import torch.nn as nn
from .inn_block import LatentINNBlock
from .rectifier import DriftRectifier

class GenMambaINN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        self.inn_blocks = nn.ModuleList([
            LatentINNBlock(config.INN_CHANNELS, config.MAMBA_D_MODEL) 
            for _ in range(config.INN_BLOCKS)
        ])
        
        self.rectifier = DriftRectifier(config.INN_CHANNELS, config.MAMBA_D_MODEL)
        
        # [创新点三] 迭代矫正步数 K
        self.rect_steps = getattr(config, 'RECT_STEPS', 3) 
        
        # [关键] 可学习的步长参数 (Learnable Step Sizes)
        # 类似于 Diffusion 的 Scheduler，让网络自己学习每一步走多远
        self.step_sizes = nn.Parameter(torch.full((self.rect_steps,), 0.5))

    def forward(self, cover, secret):
        return self.embed(cover, secret)

    def embed(self, cover, secret):
        z = torch.cat([cover, secret], dim=1) 
        for block in self.inn_blocks:
            # 1. INN 变换
            z = block(z, rev=False)
            # 2. 通道交换 (必须有，否则 Cover 永远不变)
            z = torch.cat([z.chunk(2, 1)[1], z.chunk(2, 1)[0]], dim=1)
        return z

    def extract(self, stego_damaged):
        """
        Mamba 轨迹迭代矫正 (Iterative Trajectory Rectification)
        """
        z_curr = stego_damaged
        
        # 迭代 K 步，逐步逼近真实流形
        for i in range(self.rect_steps):
            # 预测当前的漂移场 (Drift Field)
            pred_drift = self.rectifier(z_curr)
            
            # 获取当前步的可学习步长 (限制在 0~1)
            alpha = torch.sigmoid(self.step_sizes[i])
            
            # 更新状态: z_{t+1} = z_t - alpha * drift
            z_curr = z_curr - alpha * pred_drift
            
        z_rectified = z_curr
        
        # INN 逆变换
        z = z_rectified
        for block in reversed(self.inn_blocks):
            # 先撤销交换
            z = torch.cat([z.chunk(2, 1)[1], z.chunk(2, 1)[0]], dim=1)
            # 再撤销变换
            z = block(z, rev=True)
            
        return z, z_rectified