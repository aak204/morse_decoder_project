import os
import sys
import time
import argparse # Для аргументов командной строки
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Добавляем путь к src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config, utils, dataset, audio_processing, model as model_def, engine

def run_inference(model_path: str, data_dir: str, output_dir: str):
    """Запускает инференс на тестовом наборе."""
    print("--- Starting Inference ---")
    utils.seed_everything(config.SEED) # Для воспроизводимости загрузки данных
    device = config.DEVICE
    print(f"Using device: {device}")

    audio_dir = os.path.join(data_dir, 'morse_dataset', 'morse_dataset')
    test_csv_path = os.path.join(data_dir, 'test.csv')
    if not os.path.exists(test_csv_path) or not os.path.exists(audio_dir):
        print(f"Error: Test CSV or audio directory not found in {data_dir}")
        return

    print("Loading test data and vocabulary info...")
    try:
        test_df = pd.read_csv(test_csv_path)
        # Нам нужен словарь для декодера, загрузим его из трейн данных
        # В реальном проекте словарь лучше сохранять/загружать отдельно
        train_csv_path = os.path.join(data_dir, 'train.csv')
        train_df_full = pd.read_csv(train_csv_path)
        train_df_full['message'] = train_df_full['message'].apply(utils.clean_text)
        train_df_full = train_df_full[train_df_full['message'].str.len() > 0]
        char_to_int, int_to_char, blank_idx = utils.create_vocabulary(
            train_df_full['message'].tolist(), config.BLANK_CHAR
        )
        vocab_size = len(char_to_int)
        del train_df_full # Освобождаем память
    except FileNotFoundError as e:
        print(f"Error loading data for vocabulary/test set: {e}")
        return

    # Трансформ для теста
    _, eval_transform = audio_processing.get_audio_transforms()

    # Тестовый Dataset и DataLoader
    test_dataset_obj = dataset.MorseDataset(
        test_df, audio_dir, char_to_int, eval_transform, is_test=True
    )
    test_loader = DataLoader(
        test_dataset_obj, batch_size=config.INFERENCE_BATCH_SIZE, shuffle=False,
        collate_fn=dataset.collate_fn_inference, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print("Test DataLoader created.")

    print(f"Loading model architecture...")
    model = model_def.CRNN_SE_Proj(
        n_mels=config.N_MELS,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        rnn_hidden=config.RNN_HIDDEN_SIZE,
        rnn_layers=config.RNN_LAYERS,
        rnn_type=config.RNN_TYPE,
        num_heads=config.NUM_HEADS,
        num_classes=vocab_size,
        dropout=0.0, # Важно: Dropout=0 для инференса
        se_reduction=config.SE_REDUCTION
    )

    print(f"Loading model weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    try:
        # Пытаемся загрузить с weights_only для безопасности
        try: model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except: model.load_state_dict(torch.load(model_path, map_location=device)) # Fallback
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)
    model.eval() # Переводим модель в режим оценки

    predictions_map = {}
    error_ids_total = []
    pbar = tqdm(test_loader, desc="Predicting on Test Set (Greedy)")

    with torch.no_grad():
        for batch_data in pbar:
            spectrograms, spec_lengths, batch_ids, error_ids_batch = batch_data
            error_ids_total.extend(error_ids_batch) # Собираем ошибки загрузки

            if spectrograms is None: continue # Пропускаем батч с ошибками

            spectrograms = spectrograms.to(device, non_blocking=True)
            spec_lengths = spec_lengths.to(device) # Длины для модели

            try:
                log_probs, output_lengths = model(spectrograms, spec_lengths)
                batch_predictions = engine.ctc_greedy_decoder(log_probs, int_to_char, blank_idx)

                if len(batch_predictions) != len(batch_ids):
                    print(f"Warning: Prediction/ID list length mismatch.")
                    # Заполняем пустыми строками в случае несовпадения
                    for fid in batch_ids: predictions_map[fid] = ""
                else:
                    for i, fid in enumerate(batch_ids):
                        predictions_map[fid] = batch_predictions[i]

            except Exception as e:
                print(f"\nError during inference forward pass: {e}")
                # Заполняем пустыми строками для всего батча при ошибке
                for fid in batch_ids: predictions_map[fid] = ""
                if 'CUDA out of memory' in str(e):
                    print("OOM during inference. Consider reducing INFERENCE_BATCH_SIZE.")
                    # Очистка может помочь продолжить, но лучше перезапустить с меньшим батчем
                    del spectrograms, spec_lengths, log_probs, output_lengths
                    gc.collect(); torch.cuda.empty_cache()

            # Очистка в конце батча
            del spectrograms, spec_lengths
            if 'log_probs' in locals(): del log_probs
            if 'output_lengths' in locals(): del output_lengths
            if device == torch.device('cuda'): torch.cuda.empty_cache()

    # --- Создание Submission Файла ---
    print("\nGenerating submission file...")
    submission_df = test_df[['id']].copy() # Используем исходный test_df для ID
    submission_df['message'] = submission_df['id'].map(predictions_map)

    # Обработка ошибок загрузки
    if error_ids_total:
        error_ids_set = set(error_ids_total)
        print(f"Warning: {len(error_ids_set)} unique files had loading errors during inference.")
        error_ids_in_submission = error_ids_set.intersection(set(submission_df['id']))
        submission_df.loc[submission_df['id'].isin(error_ids_in_submission), 'message'] = ''

    # Обработка пропущенных предсказаний (если были ошибки forward)
    missing_preds = submission_df['message'].isnull().sum()
    if missing_preds > 0:
        print(f"Warning: {missing_preds} predictions missing (NaN). Filling with empty string.")
        submission_df['message'].fillna('', inplace=True)

    # Формируем имя выходного файла
    model_filename = os.path.basename(model_path)
    submission_filename = f"submission_{os.path.splitext(model_filename)[0]}.csv"
    submission_path = os.path.join(output_dir, submission_filename)

    try:
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved successfully to: {submission_path}")
        print("Submission DataFrame Head:")
        print(submission_df.head())
    except Exception as e:
        print(f"\nError saving submission file: {e}")

    print("--- Inference Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for Morse Code Decoder.")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the trained model state_dict (.pth file)."
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to the data directory (containing train.csv, test.csv, audio). Determined automatically if not set."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to the output directory. Determined automatically if not set."
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

    run_inference(args.model_path, data_dir, output_dir)
