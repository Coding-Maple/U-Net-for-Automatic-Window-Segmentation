#
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your custom modules
from src import config
from src.model import UNET
from src.dataset import SegmentationDataset
from src.utils import check_accuracy, save_predictions_as_imgs, CombinedLoss

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.to(device=config.DEVICE)

        with autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    print(f"Using device: {config.DEVICE}")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_IMG_DIR, exist_ok=True)

    # Augmentations setup
    train_transform = A.Compose(
        [
            A.Rotate(limit=35, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # --- SETUP MODEL, LOSS, OPTIMIZER, AND SCALER ---
    model = UNET(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES).to(config.DEVICE)
    loss_fn = CombinedLoss(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # --- (Data loader setup is the same) ---
    train_dataset = SegmentationDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=train_transform
    )
    val_dataset = SegmentationDataset(
        image_dir=config.VAL_IMG_DIR,
        mask_dir=config.VAL_MASK_DIR,
        transform=val_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False
    )

    #TRAINING LOOP
    best_dice_score = -1.0
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Validate model
        accuracy, dice_score = check_accuracy(val_loader, model, device=config.DEVICE)

        #Early stopping
        if dice_score > best_dice_score:
            best_dice_score = dice_score
            epochs_no_improve = 0
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"==> New best model saved with Dice Score: {dice_score:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Dice Score for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping triggered after {config.PATIENCE} epochs with no improvement.")
            break

        save_predictions_as_imgs(
            val_loader, model, folder=config.VALIDATION_IMG_DIR, device=config.DEVICE
        )

if __name__ == "__main__":
    main()
