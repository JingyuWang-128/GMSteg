import torch

def simulate_vae_attack(latent, noise_std=0.1):
    """
    模拟 VAE 重构带来的有损漂移 + 信道噪声
    """
    noise = torch.randn_like(latent) * noise_std
    return latent + noise

def simulate_dropout_attack(latent, drop_prob=0.1):
    """
    模拟部分 Latent 丢失 (例如裁剪或遮挡)
    """
    mask = torch.bernoulli(torch.ones_like(latent) * (1 - drop_prob))
    return latent * mask