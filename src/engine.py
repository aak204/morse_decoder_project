import torch
import torch.nn as nn
from tqdm.notebook import tqdm # или from tqdm import tqdm
import Levenshtein
import gc
from . import config # Относительный импорт

def ctc_greedy_decoder(log_probs: torch.Tensor, int_to_char: dict, blank_idx: int) -> list[str]:
    """Жадный декодер для CTC."""
    decoded_preds = []
    # Перемещаем на CPU для argmax и обработки numpy
    log_probs_cpu = log_probs.cpu()
    # argmax по измерению классов (dim=2), затем транспонируем (B, T)
    max_indices = torch.argmax(log_probs_cpu.detach(), dim=2).t().numpy()

    for indices in max_indices:
        merged_indices = []
        last_idx = -1
        for idx in indices:
            if idx != last_idx: # Схлопываем повторы
                if idx != blank_idx: # Убираем бланки
                    merged_indices.append(idx)
                last_idx = idx
        # Преобразуем индексы в символы
        text = "".join([int_to_char.get(i, '?') for i in merged_indices])
        decoded_preds.append(text)
    return decoded_preds

def calculate_levenshtein(predictions: list[str], targets: torch.Tensor,
                          target_lengths: torch.Tensor, int_to_char: dict,
                          blank_idx: int) -> tuple[int, float, list[str]]:
    """Вычисляет суммарное и среднее расстояние левенштейна."""
    total_distance = 0
    total_samples = len(predictions)
    all_true_texts = []
    targets_cpu = targets.cpu()
    target_lengths_cpu = target_lengths.cpu()

    for i in range(total_samples):
        pred_text = predictions[i]
        # Получаем истинный текст, убирая паддинг и бланки
        actual_len = min(target_lengths_cpu[i].item(), targets_cpu[i].size(0))
        target_tensor = targets_cpu[i][:actual_len]
        true_text = "".join([
            int_to_char.get(idx.item(), '?')
            for idx in target_tensor if idx.item() != blank_idx
        ])
        all_true_texts.append(true_text)

        # Расстояние Левенштейна
        distance = Levenshtein.distance(pred_text, true_text)
        total_distance += distance

    # Считаем среднее абсолютное расстояние на пример
    mean_distance_per_sample = total_distance / total_samples if total_samples > 0 else float('inf')

    return total_distance, mean_distance_per_sample, all_true_texts


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device, clip_norm: float, blank_idx: int) -> float:
    """Выполняет одну эпоху обучения."""
    model.train()
    total_loss = 0.0
    processed_batches = 0
    pbar = tqdm(dataloader, desc="Training", leave=False) # leave=False чтобы не мешать общему прогрессу

    for batch_idx, batch_data in enumerate(pbar):
        # Коллатор теперь возвращает и ошибки
        spectrograms, spec_lengths, targets, target_lengths, _, error_ids = batch_data
        if spectrograms is None: # Пропускаем батч, если он пустой после фильтрации ошибок
             print(f"Warning: Skipping empty batch {batch_idx} due to previous errors.")
             continue

        spectrograms = spectrograms.to(device, non_blocking=True)
        spec_lengths = spec_lengths.to(device) # Длины для модели
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        optimizer.zero_grad()

        try:
            log_probs, output_lengths = model(spectrograms, spec_lengths)
        except Exception as e:
            print(f"\nError forward train (batch {batch_idx}): {e}")
            optimizer.zero_grad(); continue # Пропускаем батч

        # Фильтрация для CTC Loss
        # output_lengths уже на device после модели
        valid_indices = (output_lengths > 0) & (target_lengths > 0) & (target_lengths <= output_lengths)
        if not torch.any(valid_indices):
            # print(f"Warning: No valid samples in train batch {batch_idx} after filtering.")
            optimizer.zero_grad(); continue

        log_probs_filt = log_probs[:, valid_indices, :]
        targets_filt = targets[valid_indices]
        output_lengths_filt = output_lengths[valid_indices]
        target_lengths_filt = target_lengths[valid_indices]

        if log_probs_filt.numel() == 0 or targets_filt.numel() == 0 or log_probs_filt.shape[1] == 0:
             optimizer.zero_grad(); continue

        try:
            # Длины для CTCLoss на CPU
            loss = criterion(log_probs_filt, targets_filt, output_lengths_filt.cpu(), target_lengths_filt.cpu())
        except RuntimeError as e:
            print(f"\nRuntimeError train loss (batch {batch_idx}): {e}")
            optimizer.zero_grad(); continue

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss train (batch {batch_idx}): {loss.item()}")
            optimizer.zero_grad(); continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        processed_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / processed_batches:.4f}")

        # Очистка
        del spectrograms, spec_lengths, targets, target_lengths, log_probs, output_lengths
        del log_probs_filt, targets_filt, output_lengths_filt, target_lengths_filt, loss
        if device == torch.device('cuda'): torch.cuda.empty_cache()

    return total_loss / processed_batches if processed_batches > 0 else 0.0

def validate_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                   criterion: nn.Module, device: torch.device,
                   int_to_char: dict, blank_idx: int) -> tuple[float, float]:
    """Выполняет одну эпоху валидации."""
    model.eval()
    total_loss = 0.0
    total_lev_sum = 0
    total_samples = 0
    processed_batches = 0
    all_predictions = []
    all_true_texts_val = []
    pbar = tqdm(dataloader, desc="Validating", leave=False) # leave=False

    with torch.no_grad():
        for batch_data in pbar:
            spectrograms, spec_lengths, targets, target_lengths, _, error_ids = batch_data
            if spectrograms is None: continue

            spectrograms = spectrograms.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device) # Длины для модели
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            try:
                log_probs, output_lengths = model(spectrograms, spec_lengths)
            except Exception as e:
                print(f"\nError forward val: {e}")
                continue # Пропускаем батч

            # Расчет лосса (опционально)
            valid_indices = (output_lengths > 0) & (target_lengths > 0) & (target_lengths <= output_lengths)
            loss = torch.tensor(0.0, device=device)
            if torch.any(valid_indices):
                log_probs_filt = log_probs[:, valid_indices, :]; targets_filt = targets[valid_indices]
                output_lengths_filt = output_lengths[valid_indices]; target_lengths_filt = target_lengths[valid_indices]
                if log_probs_filt.numel() > 0 and targets_filt.numel() > 0 and log_probs_filt.shape[1] > 0:
                    try:
                        batch_loss = criterion(log_probs_filt, targets_filt, output_lengths_filt.cpu(), target_lengths_filt.cpu())
                        if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)): loss = batch_loss
                    except RuntimeError: pass
            total_loss += loss.item(); processed_batches += 1

            # Декодирование
            batch_predictions = ctc_greedy_decoder(log_probs, int_to_char, blank_idx)
            all_predictions.extend(batch_predictions)

            # Метрика Левенштейна
            batch_total_dist, _, batch_true_texts = calculate_levenshtein(
                batch_predictions, targets, target_lengths, int_to_char, blank_idx
            )
            all_true_texts_val.extend(batch_true_texts)
            total_lev_sum += batch_total_dist
            total_samples += len(batch_predictions)

            pbar.set_postfix(avg_loss=f"{total_loss / max(1, processed_batches):.4f}")

            # Очистка
            del spectrograms, spec_lengths, targets, target_lengths, log_probs, output_lengths, loss
            if device == torch.device('cuda'): torch.cuda.empty_cache()

    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    # Используем абсолютное среднее расстояние на пример
    mean_lev_distance = total_lev_sum / total_samples if total_samples > 0 else float('inf')

    # Вывод результатов валидации
    print(f"\nValidation Avg Loss: {avg_loss:.4f}")
    print(f"Validation Levenshtein Mean (Per Sample): {mean_lev_distance:.4f}")
    print("Validation Examples (Greedy Predicted vs True):")
    num_examples = min(5, len(all_predictions))
    for i in range(num_examples):
        print(f"  Pred: '{all_predictions[i]}'\n  True: '{all_true_texts_val[i]}'\n  {'-'*10}")

    return avg_loss, mean_lev_distance
