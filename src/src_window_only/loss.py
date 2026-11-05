import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torch.nn.functional as F

class CombinedBinaryLoss(nn.Module):
    def __init__(self, pos_weight, dice_weight=0.5, bce_weight=0.5):
        super(CombinedBinaryLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

        # Binary Dice Loss (mode='binary')
        self.dice_loss = DiceLoss('binary')

        # Binary Cross-Entropy (BCE) with a weight for the positive class (windows)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        # BCEWithLogitsLoss expects raw logits (B, 1, H, W)
        # and targets as floats (B, 1, H, W)
        loss_bce = self.bce_loss(logits, targets.float())

        # DiceLoss expects probabilities (apply sigmoid)
        probs = F.sigmoid(logits)
        loss_dice = self.dice_loss(probs, targets)

        # Combine the losses
        return (self.dice_weight * loss_dice) + (self.bce_weight * loss_bce)
