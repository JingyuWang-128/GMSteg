# models/mamba_block.py
import torch
import torch.nn as nn

class SimplifiedMambaKernel(nn.Module):
    """
    模拟 Mamba 核心算子 (在没有安装 mamba_ssm 时使用)
    """
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: [B, L, D]
        x_and_res = self.in_proj(x)
        x_branch, res_branch = x_and_res.chunk(2, dim=-1)
        
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)
        x_branch = x_branch.transpose(1, 2)
        
        x_branch = self.act(x_branch)
        out = x_branch * self.act(res_branch) 
        return self.out_proj(out)

class FrequencyGatedSSM(nn.Module):
    """
    【创新点 2 实现】频率门控 SSM
    逻辑: Mamba 提取特征 -> 频率图生成 Gate -> 动态调制特征
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(channels, d_model)
        self.mamba = SimplifiedMambaKernel(d_model) # 如果有环境，替换为 Mamba(d_model)
        
        # 门控网络: 输入频率能量(1维) -> 输出通道门控(C维)
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, channels), 
            nn.Sigmoid() # 输出 0~1，控制嵌入强度
        )
        
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x, freq_map):
        """
        x: [B, H*W, C] 
        freq_map: [B, 1, H, W] 频率能量图
        """
        B, L, C = x.shape
        
        # 1. Mamba 特征提取
        feat = self.proj_in(x)
        feat = self.mamba(feat)
        feat_out = self.proj_out(feat)
        
        # 2. 频率门控生成 (Alignment)
        # 将空间维度的频率图展平，以对齐序列维度
        # [B, 1, H, W] -> [B, H*W, 1]
        freq_flat = freq_map.flatten(2).permute(0, 2, 1)
        
        # 生成 Gate: [B, L, 1] -> [B, L, C]
        gate = self.gate_net(freq_flat)
        
        # 3. 门控调制
        # 高频区(Gate->1): 允许 Mamba 大幅改变特征 (嵌入信息)
        # 低频区(Gate->0): 抑制 Mamba 输出 (保护原图)
        return feat_out * gate