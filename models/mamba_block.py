# models/mamba_block.py
import torch
import torch.nn as nn

# 尝试导入官方 Mamba 库，实现真正的线性复杂度 SSM
# 对应创新点 1 & 2：基于 Mamba 的高效特征提取
try:
    from mamba_ssm import Mamba
    IS_MAMBA_AVAILABLE = True
    print("[GenMamba] 检测到 'mamba_ssm' 库，启用官方 CUDA 加速核心。")
except ImportError:
    IS_MAMBA_AVAILABLE = False
    print("[GenMamba] 未检测到 'mamba_ssm'，使用 SimplifiedMambaKernel (Conv1d 模拟) 作为 fallback。")

class SimplifiedMambaKernel(nn.Module):
    """
    [Fallback] 模拟 Mamba 核心算子
    当未安装 mamba_ssm 时的替代方案，使用 Gated CNN 模拟 SSM 的部分特性。
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

def MambaLayer(d_model):
    """
    [核心修改] Mamba 层工厂函数
    统一管理 Mamba 的实例化，优先使用官方实现。
    """
    if IS_MAMBA_AVAILABLE:
        # 使用官方 Mamba，参数配置为轻量级以适应隐写任务
        return Mamba(
            d_model=d_model,
            d_state=16,  # SSM 状态维度
            d_conv=4,    # 局部卷积宽度
            expand=2     # 扩展因子
        )
    else:
        return SimplifiedMambaKernel(d_model)

class FrequencyGatedSSM(nn.Module):
    """
    [创新点 2] 频率门控状态空间模型 (FG-SSM)
    引入频率感知门控机制，根据纹理丰富度动态调整嵌入强度。
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(channels, d_model)
        
        # 使用统一的 Mamba 接口
        self.mamba = MambaLayer(d_model)
        
        # 门控网络: 输入频率能量(1维) -> 输出通道门控(C维)
        self.gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, channels), 
            nn.Sigmoid() # 输出 0~1，作为 Gate 信号
        )
        
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x, freq_map):
        """
        x: [B, H*W, C] 
        freq_map: [B, 1, H, W] 频率能量图
        """
        # 1. Mamba 特征提取 (捕捉全局依赖)
        feat = self.proj_in(x)
        feat = self.mamba(feat)
        feat_out = self.proj_out(feat)
        
        # 2. 频率门控生成 (Alignment)
        # [B, 1, H, W] -> [B, H*W, 1]
        freq_flat = freq_map.flatten(2).permute(0, 2, 1)
        
        # 生成 Gate: 根据纹理复杂度决定嵌入强度
        # High Frequency -> Gate -> 1 (允许修改)
        # Low Frequency  -> Gate -> 0 (保持原状)
        gate = self.gate_net(freq_flat)
        
        # 3. 门控调制
        return feat_out * gate