import os

# --- Model and Training Constants ---
MODEL_NAME = "facebook/wav2vec2-base"
FINE_TUNED_MODEL_PATH = "models/finetuned_kws.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Constants ---
CUSTOM_WAKE_WORD_DIR = "custom_audio/"
GSC_DATASET_PATH = "./gsc_data"
GSC_WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# --- Labels ---
NON_WAKE_WORD_LABEL = 0
WAKE_WORD_LABEL = 1

# --- Evaluation Constants ---
BATCH_SIZE = 32
