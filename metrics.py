import torch


@torch.no_grad()
def compute_iou(output, target, activation=torch.nn.Sigmoid()):
    pred = activation(output)
    inter = torch.sum(pred * target, dim=(2, 3))
    
    # Reshape directly to (batch_size, 1) due to having only the single channel
    inter = inter.view(-1, 1)
    mask_sum = torch.sum(torch.abs(target), dim=(2, 3)) + \
               torch.sum(torch.abs(pred), dim=(2, 3))
    union = mask_sum - inter
    
    smooth = .001
    return (inter + smooth) / (union + smooth)


@torch.no_grad()
def compute_dice_score(output, target, activation=torch.nn.Sigmoid()):
    pred = activation(output)
    inter = torch.sum(pred * target, dim=(2, 3))
    
    # Reshape directly to (batch_size, 1) due to having only the single channel
    inter = inter.view(-1, 1)
    mask_sum = torch.sum(torch.abs(target), dim=(2, 3)) + \
               torch.sum(torch.abs(pred), dim=(2, 3))
    mask_sum = mask_sum.view(-1, 1)
    
    smooth = .001
    return (2 * inter + smooth) / (mask_sum + smooth)
