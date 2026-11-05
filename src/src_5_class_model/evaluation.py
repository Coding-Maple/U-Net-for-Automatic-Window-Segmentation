import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import your project-specific files
from src import config
from src.model import UNET
from src.dataset import SegmentationDataset

# Configuration
TEST_IMG_DIR = ""
TEST_MASK_DIR = ""
CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

#  METRIC CONSTANTS
NUM_CLASSES_METRICS = 5
CLASS_NAMES = ["Background", "Window", "Car", "Vegetation", "Street-Lamp"]

def evaluate_model():
    print(f"Using device: {config.DEVICE}")

    #Load Test Data
    test_transform = A.Compose(
        [
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ],
    )

    test_dataset = SegmentationDataset(
        image_dir=TEST_IMG_DIR,
        mask_dir=TEST_MASK_DIR,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False
    )

    # Load Model
    # Model's output channels MUST match the 5 classes
    model = UNET(in_channels=config.IN_CHANNELS, out_channels=NUM_CLASSES_METRICS).to(config.DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded from {CHECKPOINT_PATH}")

    # Evaluation Loop
    all_preds_flat = []
    all_targets_flat = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE) # Multi-class target shape is (B, H, W)

            # PREDICTION LOGIC
            # Get logits (B, 5, H, W) and use argmax to get the final class (B, H, W)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            # Flatten and send to CPU
            all_preds_flat.append(preds.cpu().numpy().flatten())
            all_targets_flat.append(y.cpu().numpy().flatten())

    # Concatenate all batches into single giant 1D arrays
    all_preds_flat = np.concatenate(all_preds_flat).astype(np.uint8)
    all_targets_flat = np.concatenate(all_targets_flat).astype(np.uint8)

    # Calculate Metrics
    print("\nCalculating metrics...")

    # 1. Confusion Matrix (CM)
    # We explicitly tell it to expect 5 labels (0 to 4)
    cm = confusion_matrix(all_targets_flat, all_preds_flat, labels=list(range(NUM_CLASSES_METRICS)))

    # 2. Pixel Accuracy
    pixel_acc = np.diag(cm).sum() / cm.sum()

    # 3. IoU and Dice per class (Calculated from the CM)
    iou_scores = []
    dice_scores = []

    for i in range(NUM_CLASSES_METRICS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp # False Positives
        fn = cm[i, :].sum() - tp # False Negatives

        # IoU = TP / (TP + FP + FN)
        iou = (tp) / (tp + fp + fn + 1e-6)
        iou_scores.append(iou)

        # Dice = 2*TP / (2*TP + FP + FN)
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        dice_scores.append(dice)

    # 4. Mean Scores
    miou = np.mean(iou_scores)
    mdice = np.mean(dice_scores)

    # Foreground only scores (Classes 1 through 4)
    miou_foreground = np.mean(iou_scores[1:])
    mdice_foreground = np.mean(dice_scores[1:])

    # Print Results
    print("\n--- Evaluation Results ---")
    print(f"Overall Pixel Accuracy: {pixel_acc*100:.2f}%")

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<15} {'IoU':<10} {'Dice':<10}")
    print("-" * 35)

    # Print the full table of per-class metrics
    for i in range(NUM_CLASSES_METRICS):
        print(f"{CLASS_NAMES[i]:<15} {iou_scores[i]:.4f}     {dice_scores[i]:.4f}")

    print("\n--- Mean Metrics ---")
    print(f"Mean IoU (all classes):     {miou:.4f}")
    print(f"**Mean Dice (all classes):  {mdice:.4f}**")
    print("---")
    print(f"**Mean IoU (foreground):    {miou_foreground:.4f}**")
    print(f"**Mean Dice (foreground): {mdice_foreground:.4f}**")

    # --- 7. Plot and Save Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    print("Labels: ", [f"({i}) {name}" for i, name in enumerate(CLASS_NAMES)])
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Test Set (All Classes)")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "test_confusion_matrix_multiclass.png"))
    print("\nConfusion matrix plot saved to 'outputs/test_confusion_matrix_multiclass.png'")

evaluate_model()
