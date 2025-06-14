# predict.py
import os
import json
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from dataset import QuadrupletDataset # Re-use for loading test data
from utils import parse_quadruplets # For potential post-processing if needed

def predict_on_test_set(model_path, test_file_path, output_file_path):
    print(f"Using device: {config.DEVICE}")
    
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    print(f"Loading model from: {model_path}")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(config.DEVICE)
    model.eval()

    max_samples = config.MAX_SAMPLES_DEBUG if config.DEBUG_MODE else None

    print("Preparing test dataset...")
    test_dataset = QuadrupletDataset(
        file_path=test_file_path,
        tokenizer=tokenizer,
        max_source_length=config.MAX_SOURCE_LENGTH,
        max_target_length=config.MAX_TARGET_LENGTH, # Not strictly needed for generation input but good for consistency
        task_prefix=config.TASK_PREFIX,
        max_samples=max_samples,
        is_train=False # Important: test data has no 'output' field
    )

    # Use DataLoader for batching, even though QuadrupletDataset currently returns individual tensors
    # This is more for structure; actual batching happens if QuadrupletDataset returns lists of tensors
    # or if we use a custom collate_fn for test data.
    # For simplicity, let's process one by one, but batching is better for speed.
    
    # To enable batch prediction:
    # 1. Modify QuadrupletDataset.__getitem__ to return raw text or non-tensor IDs for test.
    # 2. Use a DataLoader with a custom collate_fn that tokenizes batches.
    # For now, simple loop:

    results = []
    print("Starting predictions...")
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Predicting"):
            sample = test_dataset[i]
            input_ids = sample["input_ids"].unsqueeze(0).to(config.DEVICE) # Add batch dimension
            attention_mask = sample["attention_mask"].unsqueeze(0).to(config.DEVICE) # Add batch dimension
            sample_id = sample["id"]

            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.GENERATION_MAX_LENGTH,
                num_beams=config.GENERATION_NUM_BEAMS,
                # early_stopping=True # Can be useful
            )
            
            predicted_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Ensure the output format is strictly followed
            # The model should ideally learn this, but some post-processing might be needed
            # e.g., ensuring it ends with [END] and spaces are correct.
            # For now, we assume the model learns it.
            
            results.append({"id": sample_id, "output": predicted_text})

    print(f"Saving predictions to {output_file_path}...")
    # The required submission format is a list of dicts, but usually for competition it's
    # a specific file format. Here we save as JSON list of dicts.
    # The problem description does not specify the submission file format for test,
    # but implies `output` string is what's needed. Let's adapt if there's a specific format.
    # Assuming the output for evaluation might be a file where each line is the "output" string for an ID,
    # or a JSON like the training data. The current `results` list is flexible.
    
    # If submission needs to be a JSON like train.json but with id and output:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print("Predictions saved.")

    # Optional: If ground truth for test set is available, run evaluation
    # gold_file_path = "path/to/test_ground_truth.json"
    # if os.path.exists(gold_file_path):
    #     print("Evaluating predictions...")
    #     gold_data = load_json_data(gold_file_path, max_samples=max_samples, is_train=True)
    #     gold_map = {item['id']: item['output'] for item in gold_data}
        
    #     pred_outputs_for_eval = []
    #     gold_outputs_for_eval = []
        
    #     for res_item in results:
    #         if res_item['id'] in gold_map:
    #             pred_outputs_for_eval.append(parse_quadruplets(res_item['output']))
    #             gold_outputs_for_eval.append(parse_quadruplets(gold_map[res_item['id']]))
            
    #     if pred_outputs_for_eval:
    #         scores = compute_f1_scores(pred_outputs_for_eval, gold_outputs_for_eval, config.SIMILARITY_THRESHOLD)
    #         print(f"Evaluation scores on test set: {scores}")


if __name__ == "__main__":
    # Example usage:
    # Make sure a trained model exists in config.OUTPUT_DIR + "final_model"
    trained_model_path = os.path.join(config.OUTPUT_DIR, "best_model_after_eval")
    if not os.path.exists(trained_model_path):
        print(f"Error: Trained model not found at {trained_model_path}. Please train the model first.")
    else:
        # Predict on test1.json
        predict_on_test_set(
            model_path=trained_model_path,
            test_file_path=config.TEST1_FILE,
            output_file_path=os.path.join(config.OUTPUT_DIR, "predictions_test1.json")
        )
        # Predict on test2.json (if it exists and is needed)
        if os.path.exists(config.TEST2_FILE):
             predict_on_test_set(
                model_path=trained_model_path,
                test_file_path=config.TEST2_FILE,
                output_file_path=os.path.join(config.OUTPUT_DIR, "predictions_test2.json")
            )