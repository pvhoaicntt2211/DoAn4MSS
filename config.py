# File chứa tất cả các tham số cấu hình (hyperparameters)
# file: config.py

# file: config.py

import torch

# --- Cấu hình chính ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/data"
VALID_DIR = "data/data-valid"
CHECKPOINT_DIR = "checkpoints/"

# --- Hyperparameters cho mô hình và huấn luyện ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 50
NUM_WORKERS = 0 # Đặt là 0 để tránh lỗi trên Windows, có thể tăng lên nếu dùng Linux

# --- Cấu hình xử lý âm thanh ---
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_SECONDS = 5
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)

# --- Cấu hình 4-stem separation ---
STEMS = ['vocals', 'drums', 'bass', 'other']
STEM_WEIGHTS = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}