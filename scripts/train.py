import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

# Добавляем путь к src в sys.path для импорта модулей
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src import config, utils, dataset, audio_processing, model as model_def, engine

def main():
    # Проверяем, запускаемся ли в Kaggle
    is_kaggle = os.path.exists('/kaggle/input')
    if is_kaggle:
        data_dir = '/kaggle/input/gdfag3424'
        output_dir = '/kaggle/working/'
        print("Running in Kaggle environment.")
    else:
        # Задайте свои локальные пути
        base_project_dir = project_root # Корень проекта
        data_dir = os.path.join(base_project_dir, 'data')
        output_dir = os.path.join(base_project_dir, 'output')
        print(f"Running in local environment. Data: {data_dir}, Output: {output_dir}")

    audio_dir = os.path.join(data_dir, 'morse_dataset', 'morse_dataset')
    train_csv_path = os.path.join(data_dir, 'train.csv')
    test_csv_path = os.path.join(data_dir, 'test.csv') # Не используется в обучении, но проверим наличие

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(audio_dir) or not os.path.exists(train_csv_path):
        print(f"Error: Required data not found in {data_dir}")
        return

    utils.seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    print("Loading data...")
    try:
        train_df_full = pd.read_csv(train_csv_path)
        # test_df = pd.read_csv(test_csv_path) # Загрузка тестового для инференса позже
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return
    train_df_full['message'] = train_df_full['message'].apply(utils.clean_text)
    train_df_full = train_df_full[train_df_full['message'].str.len() > 0].reset_index(drop=True)
    print(f"Train df shape after cleaning: {train_df_full.shape}")
    if train_df_full.empty: print("Error: No valid training data."); return

    char_to_int, int_to_char, blank_idx = utils.create_vocabulary(
        train_df_full['message'].tolist(), config.BLANK_CHAR
    )
    vocab_size = len(char_to_int)

    print("Splitting data...")
    train_df, val_df = train_test_split(
        train_df_full, test_size=0.15, random_state=config.SEED, shuffle=True
    )
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

    train_transform, eval_transform = audio_processing.get_audio_transforms()

    train_dataset_obj = dataset.MorseDataset(
        train_df, audio_dir, char_to_int, train_transform, is_test=False
    )
    val_dataset_obj = dataset.MorseDataset(
        val_df, audio_dir, char_to_int, eval_transform, is_test=False
    )

    print(f"Creating DataLoaders with {config.NUM_WORKERS} workers...")
    train_loader = DataLoader(
        train_dataset_obj, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=dataset.collate_fn, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset_obj, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=dataset.collate_fn, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    print("DataLoaders created.")

    print("Initializing model...")
    model = model_def.CRNN_SE_Proj(
        n_mels=config.N_MELS,
        cnn_out_channels=config.CNN_OUT_CHANNELS,
        rnn_hidden=config.RNN_HIDDEN_SIZE,
        rnn_layers=config.RNN_LAYERS,
        rnn_type=config.RNN_TYPE,
        num_heads=config.NUM_HEADS,
        num_classes=vocab_size,
        dropout=config.DROPOUT, # Dropout для обучения
        se_reduction=config.SE_REDUCTION
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized. Trainable Parameters: {total_params:,}")

    # Loss, оптимизатор, планировщик
    criterion = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True).to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=6, verbose=True
    )

    # Цикл обучения
    train_losses = []
    val_losses = []
    val_lev_distances = []
    best_val_lev = float('inf')
    epochs_no_improve = 0
    model_name_tag = f"morse_cnn_se_proj_gru_mha_h{config.RNN_HIDDEN_SIZE}_refactored"
    best_model_path = os.path.join(output_dir, f"best_{model_name_tag}_model.pth")

    print(f"\n--- Starting Training: {model_name_tag} ---")
    print(f"Saving best model to: {best_model_path}")
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{config.EPOCHS} =====")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6g}") # Формат с g

        # Обучение
        avg_train_loss = engine.train_epoch(
            model, train_loader, criterion, optimizer, device, config.CLIP_GRAD_NORM, blank_idx
        )
        # Валидация
        avg_val_loss, avg_val_lev = engine.validate_epoch(
            model, val_loader, criterion, device, int_to_char, blank_idx
        )

        # Логирование
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_lev_distances.append(avg_val_lev)
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} Summary | Time: {epoch_duration:.2f}s")
        print(f"  Avg Train Loss: {avg_train_loss:.4f}")
        # avg_val_loss и avg_val_lev уже выводятся в validate_epoch

        # Шаг планировщика
        scheduler.step(avg_val_lev)

        # Сохранение лучшей модели и Early Stopping
        if avg_val_lev < best_val_lev:
            print(f"  Validation Levenshtein improved ({best_val_lev:.4f} --> {avg_val_lev:.4f}). saving model...")
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"  model saved successfully.")
            except Exception as e:
                print(f"  Error saving model: {e}")
            best_val_lev = avg_val_lev
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation Levenshtein did not improve for {epochs_no_improve} epoch(s). best: {best_val_lev:.4f}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    total_training_time = time.time() - start_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best Validation Levenshtein Mean (Per Sample): {best_val_lev:.4f}")
    print(f"Best model weights saved at: {best_model_path}")

    if train_losses and val_losses and val_lev_distances:
        try:
            plot_path = os.path.join(output_dir, f"training_plot_{model_name_tag}.png")
            epochs_range = range(1, len(train_losses) + 1)
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, train_losses, label='Training Loss')
            plt.plot(epochs_range, val_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, val_lev_distances, label='Validation Levenshtein (Per Sample)', color='orange')
            plt.xlabel('Epochs')
            plt.ylabel('Levenshtein Distance')
            plt.title('Validation Levenshtein Distance')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(plot_path)
            print(f"Training plot saved to {plot_path}")
            plt.close()
        except Exception as e:
            print(f"Could not generate plot: {e}")
    else:
        print("No training history recorded, skipping plot generation.")

if __name__ == "__main__":
    main()
