import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from . import config # Используем относительный импорт

def get_audio_transforms() -> tuple[nn.Module, nn.Module]:
    """Возвращает трансформы для обучения и оценки."""
    eval_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS
    ).cpu() # Держим трансформы на CPU

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS
        ),
        # Применяем SpecAugment только если параметры > 0
        T.FrequencyMasking(freq_mask_param=config.FREQ_MASK_PARAM) if config.FREQ_MASK_PARAM > 0 else nn.Identity(),
        T.TimeMasking(time_mask_param=config.TIME_MASK_PARAM) if config.TIME_MASK_PARAM > 0 else nn.Identity()
    ).cpu()

    return train_transform, eval_transform
