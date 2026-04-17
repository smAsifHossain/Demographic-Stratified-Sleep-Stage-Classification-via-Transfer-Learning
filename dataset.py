import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import config as cfg


class SleepSequenceDataset(Dataset):
    """
    Dataset that yields sequences of consecutive sleep epochs.

    Each item is:
        x: (SEQ_LEN, EPOCH_SAMPLES, n_channels) float32 tensor
        y: (SEQ_LEN,) int64 tensor
    """

    def __init__(self, h5_path, subject_ids, mean=None, std=None, seq_len=None):
        self.h5_path = h5_path
        self.seq_len = seq_len or cfg.SEQ_LEN
        self.mean = mean
        self.std = std

        self.sequences = []  # list of (subject_id, start_epoch_idx)
        self._epochs = {}
        self._labels = {}

        with h5py.File(h5_path, "r") as hf:
            for subj in subject_ids:
                epochs = hf[f"{subj}/epochs"][:]
                labels = hf[f"{subj}/labels"][:]

                if mean is not None and std is not None:
                    flat = epochs.reshape(-1, epochs.shape[-1])
                    flat = (flat - mean) / std
                    epochs = flat.reshape(epochs.shape)

                self._epochs[subj] = epochs
                self._labels[subj] = labels

                n_epochs = len(labels)
                if n_epochs < self.seq_len:
                    # Pad short recordings into a single sequence
                    self.sequences.append((subj, 0))
                else:
                    for start in range(0, n_epochs - self.seq_len + 1, self.seq_len // 2):
                        self.sequences.append((subj, start))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        subj, start = self.sequences[idx]
        epochs = self._epochs[subj]
        labels = self._labels[subj]
        n_epochs = len(labels)

        end = start + self.seq_len

        if end <= n_epochs:
            x = epochs[start:end]
            y = labels[start:end]
        else:
            # Pad with zeros if recording is shorter than seq_len
            available = n_epochs - start
            pad_len = self.seq_len - available
            x = np.concatenate([
                epochs[start:n_epochs],
                np.zeros((pad_len, cfg.EPOCH_SAMPLES, epochs.shape[-1]), dtype=np.float32)
            ])
            y = np.concatenate([
                labels[start:n_epochs],
                np.full(pad_len, -1, dtype=np.int64)  # -1 = ignore in loss
            ])

        x = torch.from_numpy(x.copy()).float()
        y = torch.from_numpy(y.copy()).long()
        return x, y


class SubjectAwareDataset(Dataset):
    """
    Same as SleepSequenceDataset but also returns subject_id per sequence.
    Used during evaluation to compute per-subject metrics.
    """

    def __init__(self, h5_path, subject_ids, mean=None, std=None, seq_len=None):
        self.inner = SleepSequenceDataset(h5_path, subject_ids, mean, std, seq_len)
        self.sequences = self.inner.sequences

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x, y = self.inner[idx]
        subj, start = self.sequences[idx]
        return x, y, subj
