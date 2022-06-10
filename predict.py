import argparse
import os
import glob

import cv2
import torch
import torchvision.transforms as T

import segmentation_models_pytorch as smp
from tqdm import tqdm
import ttach as tta
import ttach.base as tta_base

from dataset import make_loader

MEAN = [0.485, 0.456, 0.406] 
STD = [0.229, 0.224, 0.225]


class SumWrapper(tta.SegmentationTTAWrapper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(len(self.transforms))
    
    def forward(self, image, *args):
        merger = tta_base.Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args).sigmoid()
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
    

@torch.no_grad()
def generate_image(model):
    model.eval()
    img_list = glob.glob(f"{args.img_root}/*.jpg")
    
    img_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size, args.img_size)),
        T.Normalize(MEAN, STD)
    ])
    
    for img_name in tqdm(img_list):
        # Load model and predict
        img = cv2.imread(img_name)
        img_t = img_transform(img)
        out = model(img_t.unsqueeze(0))
        
        # Postprocess the predict image
        if args.ttype == "sum":
            out = out[0].cpu().squeeze().numpy()
            save_img = cv2.resize(out, img.shape[:2][::-1])
            save_img = ((save_img >= args.sum_thresh) * 255).astype("uint8")
        else:
            out = out[0].sigmoid().cpu().squeeze().numpy()
            save_img = cv2.resize(out, img.shape[:2][::-1])
            save_img = ((save_img > args.thresh) * 255).astype("uint8")
        
        # Write image
        cv2.imwrite(f"{args.save_root}/{os.path.basename(img_name).split('.')[0]}.png", save_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Hyper-parameters
    parser.add_argument("--model_name", type=str, default="PAN")
    parser.add_argument("--encoder", type=str, default="timm-efficientnet-b5")
    parser.add_argument("--img_size", type=int, default=800)
    parser.add_argument("--thresh", type=float, default=0.5)
    
    # Image root
    parser.add_argument("--ttype", type=str, default="max")
    parser.add_argument("--sum_thresh", type=float, default=5.)
    parser.add_argument("--img_root", type=str, default="../Public_Image")
    parser.add_argument("--save_root", type=str, default="./predict")
    parser.add_argument("--weight", type=str, default="best.pt")
    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = getattr(smp, args.model_name)(
        encoder_name=args.encoder, encoder_weights="imagenet",
        classes=1, activation=None,
    )
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.weight))
    model = model.to(args.device)
    
    transforms = tta.Compose(
        [
            tta.Rotate90(angles=[0, 90, 180, 270]),
            tta.Scale(scales=[1, 2, 4]),
        ]
    )
    
    if args.ttype == "sum":
        tta_model = SumWrapper(model=model, transforms=tta.aliases.d4_transform(), merge_mode="sum")
    else:
        tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode=args.ttype)
    
    os.makedirs(args.save_root, exist_ok=True)
    generate_image(tta_model)
