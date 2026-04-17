import torch
import torch.nn as nn
import config as cfg


class CNNFeatureExtractor(nn.Module):
    """Per-epoch 1D-CNN that extracts a feature vector from raw multi-channel signal."""

    def __init__(self, n_channels, filters=None, kernels=None, dropout=None):
        super().__init__()
        filters = filters or cfg.CNN_FILTERS
        kernels = kernels or cfg.CNN_KERNELS
        dropout = dropout if dropout is not None else cfg.CNN_DROPOUT

        layers = []
        in_ch = n_channels
        for filt, kern in zip(filters, kernels):
            layers.extend([
                nn.Conv1d(in_ch, filt, kern, padding=kern // 2),
                nn.BatchNorm1d(filt),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ])
            in_ch = filt

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        # Compute output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, cfg.EPOCH_SAMPLES)
            out = self.conv(dummy)
            self._flat_size = out.shape[1] * out.shape[2]

        self.fc = nn.Linear(self._flat_size, cfg.CNN_FEATURE_DIM)
        self.fc_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (batch, time_samples, channels) -> need (batch, channels, time_samples)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc_relu(self.fc(x))
        return x  # (batch, CNN_FEATURE_DIM)


class SleepStageNet(nn.Module):

    def __init__(self, n_channels=7, num_classes=None, demographic_dim=0, rnn_mode="bilstm"):
        super().__init__()
        num_classes = num_classes or cfg.NUM_CLASSES
        self.rnn_mode = rnn_mode

        self.cnn = CNNFeatureExtractor(n_channels)

        lstm_input_dim = cfg.CNN_FEATURE_DIM + demographic_dim

        if rnn_mode == "bilstm":
            self.rnn = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=cfg.LSTM_HIDDEN,
                num_layers=cfg.LSTM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=cfg.LSTM_DROPOUT if cfg.LSTM_LAYERS > 1 else 0,
            )
            classifier_dim = cfg.LSTM_HIDDEN * 2

        elif rnn_mode == "lstm":
            self.rnn = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=cfg.LSTM_HIDDEN,
                num_layers=cfg.LSTM_LAYERS,
                batch_first=True,
                bidirectional=False,
                dropout=cfg.LSTM_DROPOUT if cfg.LSTM_LAYERS > 1 else 0,
            )
            classifier_dim = cfg.LSTM_HIDDEN

        elif rnn_mode == "lstm_bilstm":
            self.lstm_stage = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=cfg.LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.lstm_dropout = nn.Dropout(cfg.LSTM_DROPOUT)
            self.bilstm_stage = nn.LSTM(
                input_size=cfg.LSTM_HIDDEN,
                hidden_size=cfg.LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            classifier_dim = cfg.LSTM_HIDDEN * 2

        else:
            raise ValueError(f"Unknown rnn_mode: {rnn_mode}")

        self.classifier = nn.Linear(classifier_dim, num_classes)

    def forward(self, x, demographics=None):
        B, S, T, C = x.shape

        x_flat = x.reshape(B * S, T, C)
        features = self.cnn(x_flat)
        features = features.reshape(B, S, -1)

        if demographics is not None:
            demo_expanded = demographics.unsqueeze(1).expand(-1, S, -1)
            features = torch.cat([features, demo_expanded], dim=-1)

        if self.rnn_mode in ("bilstm", "lstm"):
            rnn_out, _ = self.rnn(features)
        elif self.rnn_mode == "lstm_bilstm":
            lstm_out, _ = self.lstm_stage(features)
            lstm_out = self.lstm_dropout(lstm_out)
            rnn_out, _ = self.bilstm_stage(lstm_out)

        logits = self.classifier(rnn_out)
        return logits


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
