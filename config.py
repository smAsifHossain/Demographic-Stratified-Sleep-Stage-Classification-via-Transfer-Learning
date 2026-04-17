import os

# Paths 
BASE_DIR = os.path.expanduser("~/Sleep_Stage_Research")
DATASET_ROOT = os.path.join(BASE_DIR, "Dataset/DREAMT_v2.1.0/2.1.0")
PROJECT_DIR = os.path.join(BASE_DIR, "conference")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
DATA_100HZ = os.path.join(DATASET_ROOT, "data_100Hz")
DATA_64HZ = os.path.join(DATASET_ROOT, "data_64Hz")
PARTICIPANT_INFO = os.path.join(DATASET_ROOT, "participant_info.csv")

PREPROCESSED_DIR = os.path.join(CHECKPOINT_DIR, "preprocessed")
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# PSG Channel Selection (7 channels) 
PSG_CHANNELS = ["C4-M1", "F4-M1", "O2-M1", "E1", "E2", "CHIN", "ECG"]

# Actual column names in CSV
PSG_CHANNEL_MAP = {
    "C4-M1": "C4-M1",
    "F4-M1": "F4-M1",
    "O2-M1": "O2-M1",
    "E1": "E1",
    "E2": "E2",
    "CHIN": "CHIN",
    "ECG": "ECG",
}

WEARABLE_CHANNELS = ["HR", "IBI", "EDA", "TEMP", "BVP", "ACC_X", "ACC_Y", "ACC_Z"]

# Sleep Stage Mapping 
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
STAGE_NAMES = ["W", "N1", "N2", "N3", "R"]
NUM_CLASSES = 5
EXCLUDE_LABELS = {"P", "Missing"}

# Signal Parameters 
SAMPLING_RATE = 100        
EPOCH_SEC = 30             
EPOCH_SAMPLES = SAMPLING_RATE * EPOCH_SEC 

# Model Hyperparameters 
SEQ_LEN = 20             
CNN_FILTERS = [64, 128, 128, 256]
CNN_KERNELS = [25, 7, 7, 7]
CNN_FEATURE_DIM = 256
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
CNN_DROPOUT = 0.5

# Training Hyperparameters 
BATCH_SIZE = 32
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
LR_PATIENCE = 5
LR_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
NUM_FOLDS = 5
VAL_RATIO = 0.1          

SEED = 42
