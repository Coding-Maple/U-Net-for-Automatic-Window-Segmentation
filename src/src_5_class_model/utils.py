import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from segmentation_models_pytorch.losses import FocalLoss
from tqdm import tqdm
from src import config




class DiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode the target mask
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2)

        # Calculate intersection and union for each class (excluding background)
        # We start from dim=1 to ignore the background class
        intersection = (probs[:, 1:] * targets_one_hot[:, 1:]).sum(dim=(2, 3))
        total = (probs[:, 1:] + targets_one_hot[:, 1:]).sum(dim=(2, 3))

        dice_score = (2. * intersection + self.epsilon) / (total + self.epsilon)

        # Average the dice score across classes and batches
        return 1. - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None, gamma=2.0):
        super(CombinedLoss, self).__init__()

        # Use the library's FocalLoss
        self.focal_loss = FocalLoss(
            mode='multiclass',
            gamma=gamma
        )

        # Custom DiceLoss
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        # FocalLoss takes logits directly
        loss_focal = self.focal_loss(logits, targets)

        # DiceLoss also takes logits (and applies softmax internally)
        loss_dice = self.dice_loss(logits, targets)

        return loss_focal + loss_dice

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    total_dice_score = 0.0 # This will be the sum of dice scores for each image
    num_classes = config.NUM_CLASSES
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device)

            preds_logits = model(x)
            preds_argmax = torch.argmax(preds_logits, dim=1) # (B, H, W)

            # Calculate Global Pixel Accuracy (as before)
            num_correct += (preds_argmax == y).sum()
            num_pixels += torch.numel(preds_argmax)

            # Calculate Mean Dice Score
            # Iterate over each image in the batch
            for i in range(x.shape[0]):
                img_pred = preds_argmax[i]
                img_true = y[i]

                # Calculate Dice for each *foreground* class
                per_class_dice = []
                for c in range(1, num_classes): # Start from 1 to exclude background (class 0)
                    pred_mask = (img_pred == c)
                    true_mask = (img_true == c)

                    # Calculate intersection (True Positives)
                    intersection = (pred_mask & true_mask).sum().item()

                    # Calculate total pixels for this class in pred and true
                    total_pred = pred_mask.sum().item()
                    total_true = true_mask.sum().item()

                    # Calculate Dice score for this class
                    # Add 1e-6 (epsilon) to prevent division by zero
                    dice = (2. * intersection) / (total_pred + total_true + 1e-6)
                    per_class_dice.append(dice)

                # Average Dice score for this single image (across all foreground classes)
                if per_class_dice:
                    total_dice_score += sum(per_class_dice) / len(per_class_dice)

    # Final Averages
    accuracy = (num_correct / num_pixels) * 100

    # Average Dice score over all images in the dataset
    avg_dice_score = total_dice_score / len(loader.dataset)

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Average Dice Score: {avg_dice_score:.4f}")

    model.train()
    return accuracy, avg_dice_score

def save_predictions_as_imgs(loader, model, folder="outputs/validation_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx >= 5:
            break
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1).unsqueeze(1)

        torchvision.utils.save_image(x, f"{folder}/input_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1).float() / (config.NUM_CLASSES - 1), f"{folder}/target_{idx}.png")
        torchvision.utils.save_image(preds.float() / (config.NUM_CLASSES - 1), f"{folder}/pred_{idx}.png")

    model.train()
