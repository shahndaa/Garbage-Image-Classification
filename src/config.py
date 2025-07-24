"""
Central configuration for the Garbage Classification project.

Edit DATA_DIR to point at your local copy of the dataset (or leave the
default, which works out of the box on Google Colab once the dataset
folder has been uploaded/mounted at that path).
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder that directly contains the 6 class sub-folders
# (cardboard/, glass/, metal/, paper/, plastic/, trash/).
# Works locally and on Colab as long as the dataset lives here.
DATA_DIR = os.environ.get("GARBAGE_DATA_DIR", os.path.join(PROJECT_ROOT, "data", "dataset-resized"))

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")  # plots, confusion matrix, etc.

MODEL_PATH = os.path.join(MODELS_DIR, "garbage_classifier.keras")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 70% train / 15% validation / 15% test, stratified by class
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
HEAD_EPOCHS = 20          # training the classification head on frozen MobileNetV2
FINE_TUNE_EPOCHS = 15     # additional epochs after unfreezing the top of the backbone
FINE_TUNE_AT_LAYER = 100  # unfreeze everything from this layer index onwards
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

DISPOSAL_GUIDE = {
    "cardboard": "Flatten boxes and remove tape or plastic labels before recycling.",
    "glass": "Rinse and remove lids/caps. Do not recycle broken window glass or ceramics.",
    "metal": "Rinse cans and remove paper labels where possible.",
    "paper": "Keep dry and remove any plastic windows or staples.",
    "plastic": "Rinse and check the resin code (1-7) printed on the item.",
    "trash": "Not recyclable through standard curbside programs — general waste bin.",
}
