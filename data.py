import os
from PIL import Image
from skimage import io
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_files, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = image_files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = io.imread(img_path)
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (np.reshape(mask, (*mask.shape, 1)) / 255).astype(np.float32)
        mask[mask > 1.0] = 1.0
        mask[mask < 0.0] = 0.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            mask = mask.permute((2, 0, 1))

        return image, mask


def get_transforms(train=True, image_height=512, image_width=512):
    train_transforms = []
    if train:
        train_transforms = [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
        ]

    transform = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            *train_transforms,
            ToTensorV2(),
        ]
    )
    return transform
