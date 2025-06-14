# split_dataset.py

import json
import random
import os
from sklearn.model_selection import train_test_split # For more robust splitting

# --- Configuration ---
ORIGINAL_TRAIN_FILE = "data/train.json"
NEW_TRAIN_FILE = "data/train_new.json"
DEV_FILE = "data/dev.json"

# Splitting strategy:
# 'ratio' for percentage-based split (e.g., 0.8 for 80% train, 20% dev)
# 'count' for a fixed number of samples in the dev set
SPLIT_MODE = 'ratio' # or 'count'
TRAIN_RATIO = 0.95  # Used if SPLIT_MODE is 'ratio'. Dev ratio will be 1.0 - TRAIN_RATIO
DEV_SET_COUNT = 1000 # Used if SPLIT_MODE is 'count'. Ensure this is less than total samples.

RANDOM_SEED = 42  # For reproducible shuffling and splitting
SHUFFLE_DATA = True # Recommended

# --- Helper Functions ---
def load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} samples from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

def save_json_data(data, file_path):
    """Saves data to a JSON file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(data)} samples to {file_path}")
    except IOError:
        print(f"Error: Could not write to file {file_path}")

# --- Main Script Logic ---
def main():
    print("Starting dataset splitting process...")

    # 1. Load original training data
    original_data = load_json_data(ORIGINAL_TRAIN_FILE)
    if original_data is None:
        return

    total_samples = len(original_data)
    print(f"Total samples in original train file: {total_samples}")

    # 2. Shuffle data (optional but recommended)
    if SHUFFLE_DATA:
        print(f"Shuffling data with random seed {RANDOM_SEED}...")
        random.seed(RANDOM_SEED)
        random.shuffle(original_data)
        print("Data shuffled.")

    # 3. Split data
    train_data = []
    dev_data = []

    if SPLIT_MODE == 'ratio':
        if not (0 < TRAIN_RATIO < 1):
            print(f"Error: TRAIN_RATIO ({TRAIN_RATIO}) must be between 0 and 1.")
            return
        
        # Using sklearn.model_selection.train_test_split for a robust split
        print(f"Splitting data with train ratio: {TRAIN_RATIO*100}%, dev ratio: {(1-TRAIN_RATIO)*100}% using scikit-learn.")
        try:
            train_data, dev_data = train_test_split(
                original_data,
                train_size=TRAIN_RATIO,
                random_state=RANDOM_SEED if SHUFFLE_DATA else None, # Pass seed if shuffling was done for reproducibility
                shuffle=False # Data is already shuffled if SHUFFLE_DATA is True
            )
        except Exception as e:
            print(f"Error during sklearn split: {e}")
            print("Falling back to simple slicing (less robust for stratification).")
            # Fallback to simple slicing if sklearn is not available or fails
            split_index = int(total_samples * TRAIN_RATIO)
            train_data = original_data[:split_index]
            dev_data = original_data[split_index:]

    elif SPLIT_MODE == 'count':
        if not (0 < DEV_SET_COUNT < total_samples):
            print(f"Error: DEV_SET_COUNT ({DEV_SET_COUNT}) must be greater than 0 and less than total samples ({total_samples}).")
            return
        print(f"Splitting data with a fixed dev set count: {DEV_SET_COUNT}")
        dev_data = original_data[:DEV_SET_COUNT] # Take the first N for dev after shuffling
        train_data = original_data[DEV_SET_COUNT:]
    else:
        print(f"Error: Invalid SPLIT_MODE '{SPLIT_MODE}'. Choose 'ratio' or 'count'.")
        return

    print(f"New training set size: {len(train_data)}")
    print(f"Development (validation) set size: {len(dev_data)}")

    # 4. Save new training set and development set
    save_json_data(train_data, NEW_TRAIN_FILE)
    save_json_data(dev_data, DEV_FILE)

    print("Dataset splitting process finished.")
    print(f"Original file '{ORIGINAL_TRAIN_FILE}' remains unchanged.")
    print(f"New training data saved to '{NEW_TRAIN_FILE}'.")
    print(f"Development data saved to '{DEV_FILE}'.")

if __name__ == "__main__":
    # Create a dummy data directory and train.json for testing if they don't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(ORIGINAL_TRAIN_FILE):
        print(f"Creating a dummy '{ORIGINAL_TRAIN_FILE}' for testing the script...")
        dummy_data = [
            {"id": i, "content": f"Sample content {i}", "output": f"Sample output {i}"}
            for i in range(100) # Create 100 dummy samples
        ]
        save_json_data(dummy_data, ORIGINAL_TRAIN_FILE)

    main()