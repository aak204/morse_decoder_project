import os
import sys
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config, utils, audio_processing

# Аугментации аудио
class AddGaussianNoise:
    def __init__(self, min_snr_db=5.0, max_snr_db=20.0, p=0.5):
        self.min_snr_db=min_snr_db
        self.max_snr_db=max_snr_db
        self.p=p
    def __call__(self, waveform):
        if random.random()<self.p:
            snr_db=random.uniform(self.min_snr_db, self.max_snr_db)
            snr=10**(snr_db/10.0)
            if waveform.ndim == 1: # Добавим поддержку 1D тензоров
                waveform = waveform.unsqueeze(0)
            _, num_samples=waveform.shape
            wf_p=waveform.norm(p=2,dim=1,keepdim=True)**2/num_samples
            n_p=wf_p/snr
            noise=torch.randn_like(waveform)*torch.sqrt(n_p)
            return (waveform+noise).squeeze(0) # Возвращаем 1D
        return waveform.squeeze(0) if waveform.ndim > 1 else waveform

class RandomTimeStretch:
    def __init__(self, factors=[0.9, 1.1], p=0.5):
        self.factors=factors
        self.p=p
    def __call__(self, waveform, sample_rate):
        if random.random()<self.p:
            rate=random.choice(self.factors)
            try:
                # F.speed ожидает (..., time)
                if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
                stretched_waveform = F.speed(waveform, sample_rate, factor=rate)[0]
                return stretched_waveform.squeeze(0) # Возвращаем 1D
            except Exception:
                return waveform.squeeze(0) if waveform.ndim > 1 else waveform
        return waveform.squeeze(0) if waveform.ndim > 1 else waveform

class RandomVolume:
    def __init__(self, factors=[0.5, 1.5], p=0.5):
        self.factors=factors
        self.p=p
    def __call__(self, waveform):
        if random.random()<self.p:
            factor=random.uniform(self.factors[0], self.factors[1])
            return waveform*factor
        return waveform

def apply_waveform_augmentations(waveform, sample_rate, augmentations):
    """Применяет waveform аугментации последовательно."""
    # Клонируем, чтобы не изменять оригинал
    waveform_aug = waveform.clone()
    stretch_transform = augmentations.get('time_stretch')
    if stretch_transform: waveform_aug = stretch_transform(waveform_aug, sample_rate)
    vol_transform = augmentations.get('volume')
    if vol_transform: waveform_aug = vol_transform(waveform_aug)
    noise_transform = augmentations.get('noise')
    if noise_transform: waveform_aug = noise_transform(waveform_aug)
    return waveform_aug

def adjust_length(waveform, target_duration_sec, sample_rate):
    """Выравнивает длину аудио до target_duration_sec."""
    target_samples = int(target_duration_sec * sample_rate) if target_duration_sec else None
    if target_samples is None: return waveform
    # Убедимся, что waveform 1D
    if waveform.ndim > 1: waveform = waveform.squeeze(0)
    num_frames = waveform.shape[0]
    if num_frames < target_samples:
        padding = (0, target_samples - num_frames)
        waveform = torch.nn.functional.pad(waveform, padding)
    elif num_frames > target_samples:
        waveform = waveform[:target_samples]
    return waveform

def plot_spectrogram(spec, title, ax, fig, db_transform=True, hop_length=config.HOP_LENGTH, sr=config.SAMPLE_RATE):
    """Рисует спектрограмму на заданных осях."""
    if spec is None:
        ax.text(0.5, 0.5, 'error loading/processing', horizontalalignment='center', verticalalignment='center', fontsize=9, color='red')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    try:
        # Преобразуем в dB, если нужно
        spec_db = T.AmplitudeToDB(top_db=80)(spec) if db_transform else spec
        # Используем librosa для отображения с осями времени/частоты
        import librosa.display # Импортируем здесь, чтобы не требовать librosa всегда
        img = librosa.display.specshow(
            spec_db.cpu().numpy(),
            sr=sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel' if spec.shape[0] == config.N_MELS else 'linear', # Определяем тип оси Y
            ax=ax,
            cmap='viridis'
        )
        fig.colorbar(img, ax=ax, format='%+.1f dB' if db_transform else '%f')
        ax.set_title(title)
    except Exception as e:
        print(f"  Error plotting {title}: {e}")
        ax.text(0.5, 0.5, 'error plotting', horizontalalignment='center', verticalalignment='center', fontsize=9, color='red')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])


def visualize(num_samples: int, data_dir: str, output_dir: str):
    """Основная функция визуализации."""
    print("--- Starting Data Visualization ---")
    utils.seed_everything(config.SEED)
    device = config.DEVICE # Не используется, но для консистентности
    print(f"Using device (for consistency): {device}")

    audio_dir = os.path.join(data_dir, 'morse_dataset', 'morse_dataset')
    train_csv_path = os.path.join(data_dir, 'train.csv')
    vis_output_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)

    if not os.path.exists(train_csv_path) or not os.path.exists(audio_dir):
        print(f"Error: Train CSV or audio directory not found in {data_dir}")
        return

    print("Loading train data...")
    try:
        train_df = pd.read_csv(train_csv_path)
        train_df['message'] = train_df['message'].apply(utils.clean_text)
        train_df = train_df[train_df['message'].str.len() > 0].reset_index(drop=True)
    except FileNotFoundError as e:
        print(f"Error loading train CSV: {e}")
        return

    print("Initializing transforms and augmentations...")
    # Mel трансформы (обучение и оценка)
    train_mel_transform, eval_mel_transform = audio_processing.get_audio_transforms()
    # Линейный трансформ
    eval_lin_transform = T.Spectrogram(
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, power=2.0
    ).cpu()

    # Waveform аугментации
    add_noise_transform = AddGaussianNoise(
        min_snr_db=config.NOISE_SNR_DB_MIN, max_snr_db=config.NOISE_SNR_DB_MAX, p=config.NOISE_PROB
    )
    time_stretch_transform = RandomTimeStretch(
        factors=config.SPEED_FACTORS, p=config.SPEED_PROB
    )
    volume_transform = RandomVolume(
        factors=config.VOL_FACTORS, p=config.VOL_PROB
    )
    train_waveform_augs = {
        'noise': add_noise_transform,
        'time_stretch': time_stretch_transform,
        'volume': volume_transform
    }

    print(f"Selecting {num_samples} random samples for visualization...")
    if num_samples > len(train_df):
        print(f"Warning: Requested {num_samples} samples, but only {len(train_df)} available. Visualizing all.")
        num_samples = len(train_df)
    random_indices = random.sample(range(len(train_df)), num_samples)

    for i, idx in enumerate(tqdm(random_indices, desc="Visualizing samples")):
        original_mel_spec = None
        augmented_mel_spec = None
        original_lin_spec = None
        augmented_lin_spec = None
        file_id = "N/A"
        row = train_df.iloc[idx]
        file_id = row['id']
        audio_path = os.path.join(audio_dir, file_id)

        try:
            # 1. Загрузка оригинала
            waveform_orig, sr_orig = torchaudio.load(audio_path)
            if sr_orig != config.SAMPLE_RATE:
                waveform_orig = T.Resample(sr_orig, config.SAMPLE_RATE)(waveform_orig)
            waveform_orig_adjusted = adjust_length(waveform_orig, config.TARGET_DURATION_SEC, config.SAMPLE_RATE)

            # 2. Оригинальные спектрограммы
            original_mel_spec = eval_mel_transform(waveform_orig_adjusted.squeeze(0))
            original_lin_spec = eval_lin_transform(waveform_orig_adjusted.squeeze(0))

            # 3. Аугментация Waveform
            waveform_aug = apply_waveform_augmentations(waveform_orig.clone(), config.SAMPLE_RATE, train_waveform_augs)
            waveform_aug_adjusted = adjust_length(waveform_aug, config.TARGET_DURATION_SEC, config.SAMPLE_RATE)

            # 4. Аугментированные спектрограммы
            augmented_mel_spec = train_mel_transform(waveform_aug_adjusted.squeeze(0))
            # Linear
            augmented_lin_spec = eval_lin_transform(waveform_aug_adjusted.squeeze(0))

        except Exception as e:
            print(f"\nError processing sample {idx} ({file_id}): {e}")
            # Оставляем None, чтобы отобразить ошибку на графике

        # 5. Построение графиков
        fig, axes = plt.subplots(2, 2, figsize=(16, 9)) # Сделал чуть менее вытянутым
        fig.suptitle(f"Sample: {file_id} (Index: {idx})")

        plot_spectrogram(original_mel_spec, "Original Mel Spectrogram (dB)", axes[0, 0], fig)
        plot_spectrogram(augmented_mel_spec, "Augmented Mel Spectrogram (dB)", axes[0, 1], fig)
        plot_spectrogram(original_lin_spec, "Original Linear Spectrogram (dB)", axes[1, 0], fig)
        plot_spectrogram(augmented_lin_spec, "Augmented Linear Spectrogram (dB)", axes[1, 1], fig)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(vis_output_dir, f"data_vis_{file_id}.png")
        plt.savefig(save_path)
        # print(f"  Saved visualization to {save_path}") # Убрал для чистоты вывода tqdm
        plt.close(fig)
        gc.collect() # Собираем мусор после каждой итерации

    print(f"\nVisualizations saved to: {vis_output_dir}")
    print("--- Visualization Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize original and augmented audio spectrograms.")
    parser.add_argument(
        "-n", "--num_samples", type=int, default=5,
        help="Number of random samples to visualize (default: 5)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to the data directory (containing train.csv, audio). Determined automatically if not set."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to the main output directory (visualizations will be saved in a 'visualizations' subfolder). Determined automatically if not set."
    )

    args = parser.parse_args()

    # Определяем пути, если не заданы
    if args.data_dir is None or args.output_dir is None:
        is_kaggle = os.path.exists('/kaggle/input')
        if is_kaggle:
            data_dir = '/kaggle/input/gdfag3424'
            output_dir = '/kaggle/working/'
        else:
            base_project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            data_dir = os.path.join(base_project_dir, 'data')
            output_dir = os.path.join(base_project_dir, 'output')
    else:
        data_dir = args.data_dir
        output_dir = args.output_dir

    visualize(args.num_samples, data_dir, output_dir)
