import os
import glob
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import config as cfg


def get_subject_list(data_dir=None):
    """Return sorted list of subject IDs from data_100Hz."""
    if data_dir is None:
        data_dir = cfg.DATA_100HZ
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    subjects = [os.path.basename(f).split("_")[0] for f in files]
    return subjects


def get_csv_path(subject_id, data_dir=None):
    """Return CSV path for a subject in data_100Hz."""
    if data_dir is None:
        data_dir = cfg.DATA_100HZ
    pattern = os.path.join(data_dir, f"{subject_id}_*.csv")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No CSV found for {subject_id} in {data_dir}")
    return matches[0]


def load_participant_info():
    """Load and return participant metadata DataFrame."""
    df = pd.read_csv(cfg.PARTICIPANT_INFO)
    df["GENDER_ENC"] = (df["GENDER"] == "M").astype(int)
    # Parse Mean_SaO2 from string like "95%" to float
    if df["Mean_SaO2"].dtype == object:
        df["Mean_SaO2_num"] = df["Mean_SaO2"].str.replace("%", "").astype(float)
    return df


def extract_epochs_from_subject(subject_id, channels=None, data_dir=None):
    """
    Load a subject CSV, extract 30-second epochs for selected channels.
    """
    if channels is None:
        channels = list(cfg.PSG_CHANNEL_MAP.values())
    csv_path = get_csv_path(subject_id, data_dir)

    cols_to_read = channels + ["Sleep_Stage"]
    df = pd.read_csv(csv_path, usecols=cols_to_read)

    signal = df[channels].values.astype(np.float32)
    stage_labels = df["Sleep_Stage"].values

    n_samples = len(signal)
    n_full_epochs = n_samples // cfg.EPOCH_SAMPLES

    signal = signal[:n_full_epochs * cfg.EPOCH_SAMPLES]
    stage_labels = stage_labels[:n_full_epochs * cfg.EPOCH_SAMPLES]

    signal = signal.reshape(n_full_epochs, cfg.EPOCH_SAMPLES, len(channels))

    # Take the label from the first sample of each epoch
    epoch_labels_str = stage_labels[::cfg.EPOCH_SAMPLES]

    # Filter out excluded labels
    valid_mask = np.array([lbl not in cfg.EXCLUDE_LABELS for lbl in epoch_labels_str])
    n_excluded = np.sum(~valid_mask)

    epochs = signal[valid_mask]
    labels = np.array([cfg.STAGE_MAP[lbl] for lbl in epoch_labels_str[valid_mask]], dtype=np.int64)

    return epochs, labels, int(n_excluded)


def preprocess_all_subjects(output_path=None, channels=None, force=False):
    """
    Extract epochs for all subjects and save to HDF5.
    Skips if output file exists unless force=True.

    HDF5 structure:
        /{subject_id}/epochs  -> (n_epochs, 3000, n_channels) float32
        /{subject_id}/labels  -> (n_epochs,) int64
    """
    if output_path is None:
        output_path = os.path.join(cfg.PREPROCESSED_DIR, "dreamt_psg7ch_epochs.h5")

    if os.path.exists(output_path) and not force:
        print(f"[SKIP] Preprocessed file exists: {output_path}")
        print(f"       Use force=True to regenerate.")
        return output_path

    if channels is None:
        channels = list(cfg.PSG_CHANNEL_MAP.values())

    subjects = get_subject_list()
    print(f"[PREPROCESS] Extracting epochs for {len(subjects)} subjects -> {output_path}")
    print(f"[PREPROCESS] Channels: {channels}")

    total_epochs = 0
    total_excluded = 0
    label_counts = np.zeros(cfg.NUM_CLASSES, dtype=np.int64)

    with h5py.File(output_path, "w") as hf:
        for subj in tqdm(subjects, desc="Extracting epochs"):
            try:
                epochs, labels, n_excl = extract_epochs_from_subject(subj, channels)
                hf.create_dataset(f"{subj}/epochs", data=epochs, compression="gzip", compression_opts=4)
                hf.create_dataset(f"{subj}/labels", data=labels)
                total_epochs += len(labels)
                total_excluded += n_excl
                for c in range(cfg.NUM_CLASSES):
                    label_counts[c] += np.sum(labels == c)
            except Exception as e:
                print(f"  [ERROR] {subj}: {e}")

    print(f"[PREPROCESS] Done. Total epochs: {total_epochs}, excluded: {total_excluded}")
    print(f"[PREPROCESS] Class distribution:")
    for i, name in enumerate(cfg.STAGE_NAMES):
        pct = 100 * label_counts[i] / total_epochs if total_epochs > 0 else 0
        print(f"  {name}: {label_counts[i]:,} ({pct:.1f}%)")

    return output_path


def load_subject_from_h5(h5_path, subject_id):
    """Load epochs and labels for a single subject from HDF5."""
    with h5py.File(h5_path, "r") as hf:
        epochs = hf[f"{subject_id}/epochs"][:]
        labels = hf[f"{subject_id}/labels"][:]
    return epochs, labels


def load_subjects_from_h5(h5_path, subject_ids):
    """Load and concatenate data for multiple subjects. Returns epochs, labels, subject_indices."""
    all_epochs = []
    all_labels = []
    all_subj_idx = []

    with h5py.File(h5_path, "r") as hf:
        for i, subj in enumerate(subject_ids):
            epochs = hf[f"{subj}/epochs"][:]
            labels = hf[f"{subj}/labels"][:]
            all_epochs.append(epochs)
            all_labels.append(labels)
            all_subj_idx.append(np.full(len(labels), i, dtype=np.int32))

    return (np.concatenate(all_epochs),
            np.concatenate(all_labels),
            np.concatenate(all_subj_idx))


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for weighted cross-entropy."""
    counts = np.bincount(labels, minlength=cfg.NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * cfg.NUM_CLASSES
    return weights.astype(np.float32)


def compute_channel_stats(h5_path, subject_ids):
    """Compute per-channel mean and std from given subjects (for normalization)."""
    running_sum = None
    running_sq_sum = None
    total_samples = 0

    with h5py.File(h5_path, "r") as hf:
        for subj in subject_ids:
            epochs = hf[f"{subj}/epochs"][:] 
            flat = epochs.reshape(-1, epochs.shape[-1])  
            if running_sum is None:
                running_sum = flat.sum(axis=0)
                running_sq_sum = (flat ** 2).sum(axis=0)
            else:
                running_sum += flat.sum(axis=0)
                running_sq_sum += (flat ** 2).sum(axis=0)
            total_samples += flat.shape[0]

    mean = running_sum / total_samples
    std = np.sqrt(running_sq_sum / total_samples - mean ** 2)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)
