import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn # Для pad_sequence
import pandas as pd
from . import config # Используем относительный импорт

class MorseDataset(Dataset):
    """Класс датасета для кода морзе."""
    def __init__(self, df: pd.DataFrame, audio_dir: str, char_map: dict,
                 audio_transform: nn.Module, is_test: bool = False):
        self.df = df
        self.audio_dir = audio_dir
        self.char_map = char_map
        self.audio_transform = audio_transform.cpu() # Трансформы на CPU
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_id = row['id']
        audio_path = os.path.join(self.audio_dir, file_id)

        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != config.SAMPLE_RATE:
                resampler = T.Resample(sr, config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Убираем batch дименшн перед transform, если он есть
            if waveform.ndim > 1 and waveform.shape[0] == 1:
                 waveform = waveform.squeeze(0)

            spectrogram = self.audio_transform(waveform) # (M, T)
            # Добавляем Channel dim для модели: (1, M, T)
            spectrogram = spectrogram.unsqueeze(0)
            spec_length = spectrogram.shape[-1]

            if not self.is_test:
                text = row['message']
                target = torch.tensor([self.char_map.get(char, -1) for char in text], dtype=torch.long) # -1 для OOV? Лучше убедиться, что все символы в словаре
                target = target[target != -1] # Убираем неизвестные символы
                target_len = torch.tensor(len(target), dtype=torch.long)
                return spectrogram, target, spec_length, target_len, file_id # Добавил ID для возможной отладки
            else:
                return spectrogram, spec_length, file_id

        except Exception as e:
            print(f"\nError loading/processing {file_id}: {e}")
            if not self.is_test:
                return None, None, None, None, file_id # Возвращаем None и ID
            else:
                return None, None, file_id # Возвращаем None и ID

# --- Collate Functions ---
def collate_fn(batch):
    """Коллатор для обучения и валидации."""
    # Фильтруем None значения (ошибки загрузки/обработки)
    valid_batch = [item for item in batch if item[0] is not None]
    error_ids = [item[4] for item in batch if item[0] is None] # Собираем ID ошибок

    if not valid_batch:
        return None, None, None, None, error_ids # Если весь батч с ошибками

    # Разбираем валидный батч
    spectrograms, targets, spec_lengths, target_lengths, ids = zip(*valid_batch)

    # Паддинг спектрограмм
    spectrograms_permuted = [s.squeeze(0).permute(1, 0) for s in spectrograms]
    spectrograms_padded = nn.utils.rnn.pad_sequence(spectrograms_permuted, batch_first=True, padding_value=0.0)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1).unsqueeze(1)

    # Тензоры длин
    spec_lengths_tensor = torch.tensor(spec_lengths, dtype=torch.long) # Используем длины *до* паддинга
    target_lengths_tensor = torch.tensor(target_lengths, dtype=torch.long)

    # Паддинг таргетов
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=config.BLANK_IDX) # Используем blank для паддинга

    return spectrograms_padded, spec_lengths_tensor, targets_padded, target_lengths_tensor, list(ids), error_ids

def collate_fn_inference(batch):
    """Коллатор для инференса (тест)."""
    valid_batch = [item for item in batch if item[0] is not None]
    error_ids = [item[2] for item in batch if item[0] is None]

    if not valid_batch:
        return None, None, error_ids

    spectrograms, spec_lengths, ids = zip(*valid_batch)

    spectrograms_permuted = [s.squeeze(0).permute(1, 0) for s in spectrograms]
    spectrograms_padded = nn.utils.rnn.pad_sequence(spectrograms_permuted, batch_first=True, padding_value=0.0)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1).unsqueeze(1)
    spec_lengths_tensor = torch.tensor(spec_lengths, dtype=torch.long)

    return spectrograms_padded, spec_lengths_tensor, list(ids), error_ids
