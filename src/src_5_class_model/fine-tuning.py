import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src import config
from src.model import UNET
from src.dataset import SegmentationDataset
from src.utils import check_accuracy, save_predictions_as_imgs, CombinedLoss


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Fine-Tuning")

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
    LOAD_CHECKPOINT = True
    CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    # Point to your new data folders
    NEW_TRAIN_IMG_DIR = ""
    NEW_TRAIN_MASK_DIR = ""
    NEW_VAL_IMG_DIR = ""
    NEW_VAL_MASK_DIR = ""

    # Use a lower learning rate for fine-tuning
    FINE_TUNE_LR = 1e-5

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

    #Normalization
    fine_tune_transform = A.Compose(
    [
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)
    # Load Model, Loss, Optimizer and Scaler
    print(f"Using device: {config.DEVICE}")
    model = UNET(in_channels=config.IN_CHANNELS, out_channels=config.NUM_CLASSES).to(config.DEVICE)
    loss_fn = CombinedLoss(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
    scaler = GradScaler()

    if LOAD_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"=> Loading checkpoint: {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(CHECKPOINT_PATH))

    else:
        print("! Warning: Checkpoint not loaded. Starting from scratch.")

    # CREATE DATA LOADERS FOR NEW DATA
    train_dataset = SegmentationDataset(
        image_dir=NEW_TRAIN_IMG_DIR,
        mask_dir=NEW_TRAIN_MASK_DIR,
        transform=train_transform
    )
    val_dataset = SegmentationDataset(
        image_dir=NEW_VAL_IMG_DIR,
        mask_dir=NEW_VAL_MASK_DIR,
        transform=fine_tune_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, shuffle=False
    )

    # TRAINING LOOP
    best_dice_score = -1.0
    epochs_no_improve = 0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        accuracy, dice_score = check_accuracy(val_loader, model, device=config.DEVICE)

        if dice_score > best_dice_score:
            best_dice_score = dice_score
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"==> New best model saved with Dice Score: {dice_score:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in Dice Score for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= config.PATIENCE:
            print(f"Early stopping triggered after {config.PATIENCE} epochs with no improvement.")
            break # Exit the training loop

        save_predictions_as_imgs(
            val_loader, model, folder=config.VALIDATION_IMG_DIR, device=config.DEVICE
        )

# ==============================================================================
# 6. RUN THE SCRIPT
# ==============================================================================
if __name__ == "__main__":
    main()
