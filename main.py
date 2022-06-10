import math
import argparse

import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from tqdm import tqdm

from dataset import make_loader
from loss import DiceLoss, FocalLoss, WeightedIOULoss
from metrics import compute_iou, compute_dice_score


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def adjust_learning_rate(optimizer, epoch):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (
            1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    progress_bar = tqdm(test_loader, desc="validation")
    dice_metrics = compute_dice_score
    
    count_img = 0
    dice_score = 0.0
    for img, mask in progress_bar:
        img, mask = img.to(args.device), mask.to(args.device)
        prediction = model.forward(img)
        
        temp_dice_score = dice_metrics(prediction, mask)
        dice_score += temp_dice_score.sum()
        count_img += img.shape[0]
        
    return dice_score / count_img


def train_one_epoch(model, epoch, train_loader, losses, optimizer):
    metrics = [
        compute_iou, compute_dice_score
    ]
    iters_per_epoch = len(train_loader)
    progress_bar = tqdm(train_loader, desc="train")
    model.train()
    for idx, (img, mask) in enumerate(progress_bar):
        adjust_learning_rate(optimizer, epoch + idx / iters_per_epoch)
        img, mask = img.to(args.device), mask.to(args.device)
        prediction = model(img)

        # Compute loss
        total_loss = 0
        for loss in losses:
            total_loss += loss(prediction, mask)
#         total_loss = structure_loss(prediction, mask)
        
        # Update model
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update metrics logs
        update_metrics = {}
        for metric_fn in metrics:
            metric_value = metric_fn(prediction, mask).cpu().mean()
            update_metrics[metric_fn.__name__] = f"{metric_value:.4f}"
            
        progress_bar.set_postfix(update_metrics)  
    

def train(model, train_laoder, test_loader, losses, optimizer):
    max_score = -1
    
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch + 1}/{args.epochs}")
        train_one_epoch(model, epoch, train_loader, losses, optimizer)
        dice_score = evaluate(model, test_loader)
        print("Validation dice score:", dice_score.item())
        
        if max_score < dice_score:
            max_score = dice_score
            torch.save(model.state_dict(), "best.pt")
            print("Model saved")
        
        print("")
        
    # Save the last model (converging state)
    torch.save(model.state_dict(), "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--encoder", type=str, default="timm-efficientnet-b5")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--lr", type=int, default=3e-4)
    parser.add_argument("--model_name", type=str, default="PAN")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    
    # Image root
    parser.add_argument("--img_root", type=str, default="../Train/images")
    parser.add_argument("--mask_root", type=str, default="../Train/masks/")
    parser.add_argument("--test_img", type=str, default="../Vali/images")
    parser.add_argument("--test_mask", type=str, default="../Vali/masks/")
    args = parser.parse_args()
    
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    model = getattr(smp, args.model_name)(
        encoder_name=args.encoder, encoder_weights="imagenet",
        classes=1, activation=None,
    )
    model = torch.nn.DataParallel(model)
    model = model.to(args.device)
    
    # Create loss function and optimizer
    dice_loss = DiceLoss()
    wiou_loss = WeightedIOULoss()
    bce_loss = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Create training and testing loader
    train_loader = make_loader(
        args.img_root, args.mask_root, args.batch_size, num_workers=16, img_size=args.img_size, mode="train")
    test_loader = make_loader(
        args.test_img, args.test_mask, args.batch_size, img_size=args.img_size, mode="val")
    train(model, train_loader, test_loader, (dice_loss, wiou_loss, bce_loss), optimizer)
    
