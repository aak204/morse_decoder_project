import os
import torch

# --- Параметры Предобработки ---
SAMPLE_RATE = 8000
N_MELS = 128
N_FFT = 400
HOP_LENGTH = 160
TARGET_DURATION_SEC = 8

# --- Параметры Модели ---
CNN_OUT_CHANNELS = 64
RNN_HIDDEN_SIZE = 512
RNN_LAYERS = 3
RNN_TYPE = 'GRU'
NUM_HEADS = 8
DROPOUT = 0.45 # Dropout для обучения
SE_REDUCTION = 16

# --- Параметры Аугментации (SpecAugment) ---
# Используются в audio_processing.py
FREQ_MASK_PARAM = 30
TIME_MASK_PARAM = 60

# --- Параметры Обучения ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 50 # Максимальное количество эпох
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
CLIP_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 10

# --- Параметры Инференса ---
INFERENCE_BATCH_SIZE = BATCH_SIZE * 2 # Можно изменить для инференса

# --- Общие Настройки ---
NUM_WORKERS = 12 # Количество воркеров для DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if DEVICE == 'cuda' else False

# --- Настройки Словаря ---
BLANK_CHAR = "_"
