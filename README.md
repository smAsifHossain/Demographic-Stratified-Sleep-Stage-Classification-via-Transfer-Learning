# Demographic-Stratified Sleep Stage Classification via Transfer Learning

This repository contains the implementation for **demographic-stratified sleep stage classification** on clinical polysomnography (PSG) using a **transfer learning** framework. The core idea is to first train a general sleep staging model on the full population and then fine-tune it for clinically meaningful demographic subgroups defined by **gender**, **age**, and **obstructive sleep apnea (OSA) severity (AHI)**.

The project is based on a **1D-CNN + BiLSTM** sequence model operating on **7-channel PSG** data and evaluates whether demographic-specific specialization improves automated sleep staging performance over a single population-agnostic baseline.

---

## Overview

Automated sleep staging is often performed using a single model trained on the entire cohort. However, sleep architecture differs across demographic groups such as gender, age, and OSA severity. This repository implements a **two-phase transfer learning pipeline**:

### Phase 1: Full-Population Pretraining
A baseline model is trained on the full cohort using **subject-wise 5-fold cross-validation**.

### Phase 2: Demographic Subgroup Fine-Tuning
The pretrained model is then fine-tuned independently on subgroup-specific data for:

- Gender
- Age group
- AHI severity
- Gender × AHI
- Gender × Age
- Age × AHI

This design enables subgroup adaptation while preserving the benefits of full-population pretraining.

---

## Key Highlights

- **7-channel clinical PSG** input
- **1D-CNN + BiLSTM** architecture for spatial-temporal sleep stage modeling
- **Sequence-based learning** over consecutive epochs
- **Two-phase transfer learning** with strict leakage prevention
- **Subject-wise cross-validation**
- **Demographic-aware fine-tuning** across 37 subgroup configurations
- Evaluation using:
  - Accuracy
  - Macro F1-score
  - Cohen’s Kappa
  - Per-class F1-scores

---

## Project Structure

```text
.
├── config.py                             # All paths, hyperparameters, channel definitions
├── data_utils.py                         # CSV loading, epoch extraction, HDF5 I/O, normalization
├── dataset.py                            # PyTorch Dataset classes (sequence-based)
├── model.py                              # 1D-CNN + BiLSTM architecture
├── trainer.py                            # Training loop with checkpointing and resumability
├── eval_utils.py                         # Metrics, confusion matrices, plots

├── exp0_preprocess.ipynb                 # Run once: extract epochs and save as HDF5
├── exp0_CNN+BiLSTM_train.ipynb           # Baseline architecture experiment
├── exp0_1_CNN+LSTM+BiLSTM_train.ipynb    # Architecture comparison experiment
├── exp0_2_CNN+LSTM_train.ipynb           # Architecture comparison experiment

├── exp2_1_gender_finetuned_train.ipynb   # Gender-based fine-tuning
├── exp3_1_age_finetuned_train.ipynb      # Age-based fine-tuning
├── exp4_1_ahi_finetuned_train.ipynb      # AHI-based fine-tuning

├── exp5_1_gender_ahi_finetuned_train.ipynb   # Gender × AHI fine-tuning
├── exp5_2_gender_age_finetuned_train.ipynb   # Gender × Age fine-tuning
├── exp5_3_age_ahi_finetuned_train.ipynb      # Age × AHI fine-tuning

└── README.md
