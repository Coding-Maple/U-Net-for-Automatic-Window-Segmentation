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

# For our metrics, we have 2 classes (0=Background, 1=Window)
NUM_CLASSES_METRICS = 2
CLASS_NAMES = ["Background", "Window"]

def evaluate_model():
    print(f"Using device: {config.DEVICE}")

    # Load test data
    # We only need normalization for the test set
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
        shuffle=False # Never shuffle a test set
    )

    # Load Model
    model = UNET(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES).to(config.DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded from {CHECKPOINT_PATH}")

    # Evaluation Loop
    # We will accumulate all predictions and targets to calculate metrics
    all_preds_flat = []
    all_targets_flat = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE) # Shape (B, 1, H, W)
            
            # Get predictions
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5) # Shape (B, 1, H, W)

            # Flatten and send to CPU
            all_preds_flat.append(preds.cpu().numpy().flatten())
            all_targets_flat.append(y.cpu().numpy().flatten())

    # Concatenate all batches into single giant 1D arrays
    all_preds_flat = np.concatenate(all_preds_flat).astype(np.uint8)
    all_targets_flat = np.concatenate(all_targets_flat).astype(np.uint8)

    # Calculate Metrics
    print("\nCalculating metrics...")

    # Confusion Matrix
    cm = confusion_matrix(all_targets_flat, all_preds_flat, labels=[0, 1])
    
    # Pixel Accuracy
    pixel_acc = np.diag(cm).sum() / cm.sum()
    
    # From the confusion matrix, calculate TP, FP, FN for each class
    # Class 0 (Background)
    tp_bg = cm[0, 0]
    fp_bg = cm[1, 0]
    fn_bg = cm[0, 1]
    
    # Class 1 (Window)
    tp_win = cm[1, 1]
    fp_win = cm[0, 1]
    fn_win = cm[1, 0]

    # Intersection over Union (IoU) per class
    # IoU = TP / (TP + FP + FN)
    iou_bg = (tp_bg) / (tp_bg + fp_bg + fn_bg + 1e-6)
    iou_win = (tp_win) / (tp_win + fp_win + fn_win + 1e-6)
    
    # Mean Intersection over Union (mIoU)
    miou = (iou_bg + iou_win) / NUM_CLASSES_METRICS

    # Dice Coefficient per class
    # Dice = 2*TP / (2*TP + FP + FN)
    dice_bg = (2 * tp_bg) / (2 * tp_bg + fp_bg + fn_bg + 1e-6)
    dice_win = (2 * tp_win) / (2 * tp_win + fp_win + fn_win + 1e-6)

    # --- 6. Print Results ---
    print("\n--- Evaluation Results ---")
    print(f"Overall Pixel Accuracy: {pixel_acc*100:.2f}%")
    
    print("\n--- Intersection over Union (IoU) ---")
    print(f"  IoU (Background): {iou_bg:.4f}")
    print(f"  IoU (Window):     {iou_win:.4f}")
    print(f"Mean IoU (mIoU):      {miou:.4f}")
    
    print("\n--- Dice Coefficient ---")
    print(f"  Dice (Background): {dice_bg:.4f}")
    print(f"  Dice (Window):     {dice_win:.4f}")
    
    print("\n--- Confusion Matrix ---")
    print(f"{'':<12} {'Pred Background':<15} {'Pred Window':<15}")
    print(f"{'True Background':<12} {cm[0, 0]:<15} {cm[0, 1]:<15}")
    print(f"{'True Window':<12} {cm[1, 0]:<15} {cm[1, 1]:<15}")

    # Plot and Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix for Test Set")
    
    # Create outputs dir if it doesn't exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "test_confusion_matrix.png"))
    print("\nConfusion matrix plot saved to 'outputs/test_confusion_matrix.png'")

evaluate_model()
