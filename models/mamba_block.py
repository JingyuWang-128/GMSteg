import torch
import torch.nn as nn

# 尝试导入官方 Mamba，提供回退方案
try:
    from mamba_ssm import Mamba
except ImportError:
    print("❌ Warning: 'mamba_ssm' not found. Using a dummy Mamba placeholder. (Please install mamba-ssm!)")
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model * expand),
                nn.SiLU(),
                nn.Linear(d_model * expand, d_model)
            )
        def forward(self, x): return self.net(x)

class SemanticTextureGatedMamba(nn.Module):
    """
    [创新点二] 语义与纹理双重门控 Mamba
    """
    def __init__(self, channels, d_model=64):
        super().__init__()
        self.proj_in = nn.Linear(channels, d_model)
        
        # 使用官方 Mamba 模块进行长序列建模
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=16,  
            d_conv=4,    
            expand=2     
        )
        
        # 1. 纹理门控 (Texture Gate) - 基于频率输入
        self.texture_gate_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, channels), 
            nn.Sigmoid()
        )

        # 2. 语义门控 (Semantic Gate) - 基于 Mamba 深层语义特征
        self.semantic_gate_net = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.SiLU(),
            nn.Linear(16, channels),
            nn.Sigmoid() 
        )
        
        self.proj_out = nn.Linear(d_model, channels)

    def forward(self, x, freq_map):
        """
        x: [B, L, C]
        freq_map: [B, 1, H, W]
        """
        # Mamba 提取特征 (捕捉长程语义依赖)
        feat = self.proj_in(x)
        mamba_feat = self.mamba(feat) # [B, L, D]
        
        # --- 计算双重门控 ---
        
        # A. 纹理分数: 频率越高 -> 纹理越复杂 -> 容量越大
        freq_flat = freq_map.flatten(2).permute(0, 2, 1) # [B, L, 1]
        score_texture = self.texture_gate_net(freq_flat)
        
        # B. 语义重要性: Mamba 激活强的地方通常是语义核心 -> 敏感度高
        importance_semantic = self.semantic_gate_net(mamba_feat)
        
        # C. 融合策略
        # 第一标准：语义 (严控敏感区)
        # 第二标准：纹理 (决定嵌入量)
        # beta 控制对语义的避让程度，beta 越大越保守
        beta = 2.0 
        final_gate = score_texture * torch.pow((1.0 - importance_semantic), beta)
        
        feat_out = self.proj_out(mamba_feat)
        
        # 门控调制：只在 (纹理丰富 AND 非语义核心) 的地方激活特征
        return feat_out * final_gate