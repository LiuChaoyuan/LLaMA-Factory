# config.py
import torch

# --- General Settings ---
DEBUG_MODE = False  # Set to False for full dataset run
MAX_SAMPLES_DEBUG = 100 # Number of samples to use in debug mode
SEED = 42

# --- File Paths ---
TRAIN_FILE = "data/train_new.json"
DEV_FILE = "data/dev.json" # 验证集文件路径
TEST1_FILE = "data/test1.json"
TEST2_FILE = "data/test2.json"
OUTPUT_DIR = "/root/autodl-tmp/NLP_results/"
PREDICTION_FILE = OUTPUT_DIR + "predictions.json"
EVAL_RESULTS_FILE = OUTPUT_DIR + "eval_results.txt" # 保存评估结果的文件


# --- Model & Tokenizer ---
MODEL_NAME = "mengzi-t5-base" # Or other suitable Chinese T5/BART
TASK_PREFIX = "抽取仇恨言论四元组："

# --- Training Hyperparameters ---
NUM_TRAIN_EPOCHS = 20 # Start with 3-5, adjust based on validation
PER_DEVICE_TRAIN_BATCH_SIZE = 32 # Adjust based on GPU memory
PER_DEVICE_EVAL_BATCH_SIZE = 32 # 假设你用这个作为预测批次大小
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 2
MAX_SOURCE_LENGTH = 512 # Max length for input text
MAX_TARGET_LENGTH = 128 # Max length for generated output (quadruplets)
GENERATION_MAX_LENGTH = 128 # Ensure this is same or more than MAX_TARGET_LENGTH
GENERATION_NUM_BEAMS = 3
LOAD_BEST_MODEL_AT_END = True # <--- 新增或修改

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluation ---
SIMILARITY_THRESHOLD = 0.5