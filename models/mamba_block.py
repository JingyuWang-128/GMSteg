import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedMambaKernel(nn.Module):
    """
    为了演示方便，使用 PyTorch 原生算子模拟 Mamba 的核心逻辑。
    正式使用时请替换为 `mamba_ssm` 的官方实现。
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
        
        # 1. 卷积分支 (模拟 SSM 局部性)
        x_branch = x_branch.transpose(1, 2) # [B, D, L]
        x_branch = self.conv1d(x_branch)
        x_branch = x_branch.transpose(1, 2)
        
        # 2. 激活与门控
        x_branch = self.act(x_branch)
        # 3. 模拟 SSM 状态更新 (这里用 Gating 简化替代)
        out = x_branch * self.act(res_branch) 
        
        return self.out_proj(out)

class FrequencyGatedSSM(nn.Module):
    """
    [Innovation 2] 频率门控 SSM
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        # 特征投影
        self.proj_in = nn.Linear(channels, d_model)
        
        # Mamba 核心
        self.mamba = SimplifiedMambaKernel(d_model)
        
        # 频率感知门控网络
        # 输入: 频率能量值 (scalar) -> 输出: 通道门控权重
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, channels), # 为每个通道生成不同的门控
            nn.Sigmoid() 
        )
        
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x, freq_map):
        """
        x: [B, H*W, C] 序列化潜码
        freq_map: [B, 1, H, W] 频率能量图
        """
        B, L, C = x.shape
        
        # 1. Mamba 提取全局上下文特征
        feat = self.proj_in(x)
        feat_mamba = self.mamba(feat)
        feat_out = self.proj_out(feat_mamba)
        
        # 2. 生成频率门控 (Frequency Gating)
        # 将频率图展平以对齐序列 [B, H*W, 1]
        freq_flat = freq_map.view(B, L, 1)
        # 生成 Gate: [B, L, C]
        gate = self.gate_net(freq_flat)
        
        # 3. 动态调制
        # 策略: Gate 趋向 1 (高频区) -> 允许大幅修改(嵌入)
        #       Gate 趋向 0 (低频区) -> 抑制修改(保护)
        return feat_out * gate