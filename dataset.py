import glob

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import cv2
from PIL import Image
import albumentations as A

MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]


class SegmentationDataset(Dataset):
    def __init__(self, img_root, mask_root=None, img_size=224,
                 mode="train", transforms=None):
        self.img_list =glob.glob(f"{img_root}/*.jpg") + glob.glob(f"{img_root}/*.png")
        self.img_root = img_root
        self.mask_root = mask_root
        
        self.mask_transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.Resize((img_size, img_size)),
        ])
        
        self.total_transform = A.Compose([
            # Spatial Transformation
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),
            
            # Blurring image or adding noise
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),           
        
            # Color transformation
            A.OneOf([                        
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
                A.CLAHE()
            ], p=0.5),
            
            A.HueSaturationValue(p=0.3),
        ])
        
        self.mode = mode
        if transforms is not None:
            self.img_transform = transforms
        else:
            self.img_transform = T.Compose([
                T.ToTensor(),
                T.Resize((img_size, img_size)),
                T.Normalize(MEAN, STD)
            ])
            
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        image = cv2.imread(image_path)
        
        if self.mode == "train":
            mask_path = image_path.replace(self.img_root, self.mask_root)
            mask = cv2.imread(mask_path)
            transform = self.total_transform(image=image, mask=mask)            
            image = self.img_transform(transform["image"])
            mask = self.mask_transform(transform["mask"])
            
            return image, mask
        elif self.mode == "val":
            mask_path = image_path.replace(self.img_root, self.mask_root)
            mask = cv2.imread(mask_path)
            image = self.img_transform(image)
            mask = self.mask_transform(mask)
                                
            return image, mask
        else:
            image = self.img_transform(image)
            return image, image_path
    
    
def make_loader(image_root, mask_root, batch_size, 
                img_size=384, shuffle=True, num_workers=16, 
                mode="train", pin_memory=True):

    dataset = SegmentationDataset(
        image_root, mask_root, mode=mode, img_size=img_size)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=True)
    
    return data_loader
    