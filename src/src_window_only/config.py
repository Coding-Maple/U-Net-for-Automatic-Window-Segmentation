import torch

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 # Keep this high for a powerful GPU
NUM_EPOCHS = 100 # Increased for longer training
NUM_WORKERS = 8
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024
PIN_MEMORY = True
PATIENCE = 5 # Number of epochs to wait for improvement before stopping

NUM_CLASSES = 2
IN_CHANNELS = 1

# --- Paths ---
DATA_DIR = ""
TRAIN_IMG_DIR = DATA_DIR + ""
TRAIN_MASK_DIR = DATA_DIR + ""
VAL_IMG_DIR = DATA_DIR + ""
VAL_MASK_DIR = DATA_DIR + ""

OUTPUT_DIR = "outputs/"
CHECKPOINT_DIR = OUTPUT_DIR + ""
VALIDATION_IMG_DIR = OUTPUT_DIR + ""
