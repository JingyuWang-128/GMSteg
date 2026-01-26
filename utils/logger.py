import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
import sys

def get_logger(name, save_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Stream Handler (控制台)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # File Handler (文件)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, 'train.log'), mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger