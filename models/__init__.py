# models/__init__.py

# 从各个子文件中导入核心类
from .gen_mamba import GenMambaINN
from .inn_block import LatentINNBlock
from .mamba_block import SemanticTextureGatedMamba
from .rectifier import DriftRectifier

# 定义当使用 from models import * 时，哪些类会被导入
__all__ = [
    'GenMambaINN', 
    'LatentINNBlock', 
    'FrequencyGatedSSM', 
    'DriftRectifier'
]