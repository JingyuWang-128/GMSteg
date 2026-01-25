# utils/datasets.py
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, recursive=True):
        """
        兼容扁平目录(DIV2K)和层级目录(ImageNet)的通用数据集读取类
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # 支持常见图片格式
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp')
        self.image_paths = []
        
        # 递归或非递归查找所有图片
        if recursive:
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        else:
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(root_dir, ext), recursive=False))
                
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {root_dir} 下未找到任何图片，请检查路径配置！")
            
        print(f"Dataset loaded from {root_dir}: {len(self.image_paths)} images found.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0 # 返回 0 作为 dummy label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 简单的错误处理：随机返回另一张图
            return self.__getitem__((idx + 1) % len(self))

def get_dataloader(data_path, image_size, batch_size, is_train=True):
    if is_train:
        # 训练时：随机裁剪 (RandomCrop) 增加数据多样性
        transform = transforms.Compose([
            transforms.RandomCrop(image_size, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        # 验证/测试时：中心裁剪 (CenterCrop) 或 Resize 保证结果确定性
        transform = transforms.Compose([
            transforms.Resize(image_size), # 或者 CenterCrop
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    dataset = CustomImageDataset(data_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, 
                      num_workers=4, drop_last=is_train)