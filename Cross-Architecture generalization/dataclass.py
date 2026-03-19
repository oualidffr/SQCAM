import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from PIL import Image

class ArrowSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.to_tensor = transforms.ToTensor()
        self.image_ids = [f.rsplit('.', 1)[0] for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]  # ISIC
        #self.image_ids = [f.rsplit('.', 1)[0] for f in os.listdir(image_dir) if f.lower().endswith('.bmp')] # PH2
        self.images = []
        self.masks = []

        for img_id in self.image_ids:
            # ISIC
            img_path = os.path.join(image_dir, img_id + ".jpg")
            mask_path = os.path.join(mask_dir, img_id + "_segmentation.png")
            #PH2
            #img_path = os.path.join(image_dir, img_id + ".bmp")
            #mask_path = os.path.join(mask_dir, img_id + ".bmp")

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            self.images.append(np.array(image))
            self.masks.append(np.array(mask))

        # Augmentations 
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        ])
        print("Loading resized Data into memory done.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_np = self.images[idx]
        mask_np = self.masks[idx]
        img_id = self.image_ids[idx]
        
        if self.is_train:
            transformed = self.train_transform(image=image_np, mask=mask_np)
            image_np = transformed["image"]
            mask_np = transformed["mask"]

        image_tensor = self.to_tensor(image_np)  
        mask_tensor = self.to_tensor(mask_np)
        mask_tensor = (mask_tensor > 0).float()    

        return image_tensor, mask_tensor, img_id