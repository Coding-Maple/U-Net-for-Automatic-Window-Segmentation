import torch
import torchvision
from tqdm import tqdm
from src import config

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device) # Shape is (B, 1, H, W)

            #Binary Logic
            # Get raw logits from the model
            logits = model(x)
            # Apply sigmoid and threshold to get predictions
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float() # Convert to 0s and 1s

            #Calculate Pixel Accuracy
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            #Calculate Dice Score
            # (2 * TP) / (2 * TP + FP + FN)
            intersection = (preds * y).sum()
            total_sum = preds.sum() + y.sum()
            dice_score += (2. * intersection + 1e-6) / (total_sum + 1e-6)

    accuracy = (num_correct / num_pixels) * 100
    # Average dice score across all batches
    avg_dice_score = dice_score / len(loader)

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Average Dice Score: {avg_dice_score:.4f}")

    model.train()
    return accuracy, avg_dice_score

def save_predictions_as_imgs(loader, model, folder="outputs/validation_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        if idx >= 5: # Save first 5 images
            break
        x = x.to(device=device)
        y = y.to(device)

        with torch.no_grad():
            # Binary Logic
            logits = model(x)
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()

        # Save input, ground truth, and prediction
        # No need for .unsqueeze(1) as 'y' and 'preds' are already (B, 1, H, W)
        torchvision.utils.save_image(x, f"{folder}/input_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}/target_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

    model.train()

