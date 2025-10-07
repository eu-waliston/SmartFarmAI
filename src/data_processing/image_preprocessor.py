import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AdvancedImagePreprocessor:
    def __init__(self, config):
        self.config = config
        self.train_transform = A.Compose([
            A.Resize(config['model']['input_size'], config['model']['input_size']),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=config['data']['image_transform']['mean'],
                std=config['data']['image_transform']['std']
            ),
            ToTensorV2(),
        ])

        self.val_transform = A.Compose([
            A.Resize(config['model']['input_size'], config['model']['input_size']),
            A.Normalize(
                mean=config['data']['image_transform']['mean'],
                std=config['data']['image_transform']['std']
            ),
            ToTensorV2(),
        ])

    def apply_train_transform(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.train_transform(image=image)['image']

    def apply_val_transform(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        return self.val_transform(image=image)['image']