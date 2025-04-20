import random
import os
import torch
import numpy as np
from collections import Counter

def seed_everything(seed: int):
    """Устанавливает seed для воспроизводимости."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        # Следующие строки могут замедлить обучение, но нужны для полной воспроизводимости
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def clean_text(text: str) -> str:
    """Очищает текст от лишних пробелов."""
    if not isinstance(text, str): text = str(text)
    return text.strip()

def create_vocabulary(texts: list[str], blank_char: str = "_") -> tuple[dict, dict, int]:
    """Создает словарь символов и возвращает маппинги."""
    all_chars = Counter("".join(texts))
    vocab_list = sorted(all_chars.keys()) + [blank_char]
    vocab_size = len(vocab_list)
    char_to_int = {char: i for i, char in enumerate(vocab_list)}
    int_to_char = {i: char for i, char in enumerate(vocab_list)}
    blank_idx = char_to_int[blank_char]
    print(f"\nVocabulary: {''.join(vocab_list)}")
    print(f"Vocabulary size (incl. blank): {vocab_size}")
    return char_to_int, int_to_char, blank_idx
