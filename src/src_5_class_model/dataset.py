import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Assumes mask has same name but .png extension
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        # Load mask as a standard numpy array first
        mask = np.array(Image.open(mask_path).convert("L"))

        # Add the channel dimension at the end for a (H, W, C) format for Albumentations
        image = np.expand_dims(image, axis=-1)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert mask to the required torch.long data type
        mask_tensor = mask.to(torch.long)

        return image, mask_tensor

