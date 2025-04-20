import torch
import torch.nn as nn
from . import config # Относительный импорт

class SEBlock(nn.Module):
    """Блок Squeeze-and-Excitation."""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CRNN_SE_Proj(nn.Module):
    """Основная архитектура модели CRNN с SE, проекцией и MHA."""
    def __init__(self, n_mels: int, cnn_out_channels: int, rnn_hidden: int,
                 rnn_layers: int, rnn_type: str, num_heads: int,
                 num_classes: int, dropout: float, se_reduction: int):
        super().__init__()
        self.n_mels = n_mels
        self.blank_idx = config.BLANK_IDX # Получаем из конфига

        cnn_ch1 = cnn_out_channels
        cnn_ch2 = cnn_out_channels * 2

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_ch1, 3, 1, 1), nn.BatchNorm2d(cnn_ch1), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(cnn_ch1, cnn_ch1, 3, 1, 1), nn.BatchNorm2d(cnn_ch1), nn.GELU(),
            SEBlock(cnn_ch1, se_reduction), nn.MaxPool2d((2, 1)),
            nn.Conv2d(cnn_ch1, cnn_ch2, 3, 1, 1), nn.BatchNorm2d(cnn_ch2), nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(cnn_ch2, cnn_ch2, 3, 1, 1), nn.BatchNorm2d(cnn_ch2), nn.GELU(),
            SEBlock(cnn_ch2, se_reduction), nn.MaxPool2d((2, 1))
        )

        # Рассчитываем выходные фичи CNN динамически
        with torch.no_grad():
            # Используем примерную длину из конфига, если задана
            dummy_time = int(config.TARGET_DURATION_SEC * config.SAMPLE_RATE / config.HOP_LENGTH) + 1 if hasattr(config, 'TARGET_DURATION_SEC') and config.TARGET_DURATION_SEC else 400
            dummy_input = torch.randn(1, 1, n_mels, dummy_time)
            cnn_out = self.cnn(dummy_input)
            self.cnn_output_features = cnn_out.shape[1] * cnn_out.shape[2] # C_out * H_out
            # Рассчитываем фактор уменьшения времени CNN
            self.cnn_time_reduction_factor = dummy_input.shape[3] / cnn_out.shape[3]
            print(f"CNN output features calculated: {self.cnn_output_features}")
            print(f"CNN time reduction factor: {self.cnn_time_reduction_factor:.2f}")


        self.projection_dim = rnn_hidden * 2
        self.cnn_to_rnn_proj = nn.Linear(self.cnn_output_features, self.projection_dim)
        self.cnn_to_rnn_act = nn.GELU()
        print(f"Projection from {self.cnn_output_features} to {self.projection_dim}")

        rnn_class = nn.GRU if rnn_type.upper() == 'GRU' else nn.LSTM
        self.rnn = rnn_class(
            self.projection_dim, rnn_hidden, rnn_layers,
            bidirectional=True, dropout=dropout if rnn_layers > 1 else 0, batch_first=True
        )

        self.mha_embed_dim = rnn_hidden * 2
        if self.mha_embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({self.mha_embed_dim}) must be divisible by num_heads ({num_heads})")

        self.mha = nn.MultiheadAttention(self.mha_embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_mha = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.mha_embed_dim)
        self.fc = nn.Linear(self.mha_embed_dim, num_classes)

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Рассчитывает длины выхода после CNN."""
        # input_lengths - это длины спектрограмм (T измерение)
        output_lengths = torch.floor(input_lengths.float() / self.cnn_time_reduction_factor)
        # Убедимся, что минимальная длина 1
        return torch.clamp(output_lengths.long(), min=1)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 1, M, T)
        # input_lengths: (B,) - длины T до паддинга
        x = self.cnn(x) # (B, C_out, H_out, T_cnn)
        batch_size, _, _, time_cnn_padded = x.shape

        # (B, T_cnn, C_out, H_out) -> (B, T_cnn, C_out * H_out)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, time_cnn_padded, self.cnn_output_features)

        x = self.cnn_to_rnn_proj(x)
        x = self.cnn_to_rnn_act(x) # (B, T_cnn, projection_dim)

        output_lengths = self._get_output_lengths(input_lengths)

        # Проверка на нулевые длины *после* CNN
        if torch.any(output_lengths <= 0):
            print(f"Warning: Zero/negative output lengths after CNN: {output_lengths.tolist()}. Input lengths: {input_lengths.tolist()}")
            # Возвращаем пустой выход с высокой вероятностью blank
            zero_log_probs = torch.full((time_cnn_padded, batch_size, self.fc.out_features), -float('inf'), device=x.device, dtype=x.dtype)
            zero_log_probs[..., self.blank_idx] = 0 # log(1)=0
            # Возвращаем обрезанные до 0 длины
            return zero_log_probs, torch.clamp(output_lengths, min=0)

        # Упаковка последовательности (длины на CPU)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, output_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        self.rnn.flatten_parameters() # Оптимизация для DataParallel/CUDA
        packed_output, _ = self.rnn(packed_input) # (PackedSequence)

        # Распаковка (с указанием общей длины = time_cnn_padded)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=time_cnn_padded
        ) # (B, T_cnn, rnn_hidden * 2)

        # Multi-Head Attention
        max_len_rnn = gru_output.size(1) # == time_cnn_padded
        idx = torch.arange(max_len_rnn, device=x.device).unsqueeze(0) # (1, T_cnn)
        # True где нужно маскировать (за пределами output_lengths)
        key_padding_mask = (idx >= output_lengths.unsqueeze(1)) # (B, T_cnn)

        attn_output, _ = self.mha(gru_output, gru_output, gru_output, key_padding_mask=key_padding_mask)

        # Residual + LayerNorm
        x = gru_output + self.dropout_mha(attn_output)
        x = self.layer_norm(x)

        # Финальный слой и LogSoftmax для CTC
        x = self.fc(x) # (B, T_cnn, num_classes)
        # Нужен формат (T, B, C) для CTCLoss
        log_probs = nn.functional.log_softmax(x.permute(1, 0, 2), dim=2)

        return log_probs, output_lengths
