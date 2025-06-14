# train.py
import os
import torch
import numpy as np
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer, # Use Fast Tokenizer
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from dataset import QuadrupletDataset
import config  # Import config from config.py
from utils import parse_quadruplets, compute_f1_scores # For custom evaluation
import json

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed) # If using Python's random module

def train_model():
    set_seed(config.SEED)

    print(f"Using device: {config.DEVICE}")
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
    
    print(f"Loading model: {config.MODEL_NAME}")
    model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
    model.to(config.DEVICE)

    max_samples = config.MAX_SAMPLES_DEBUG if config.DEBUG_MODE else None
        # For dev set, you might want to use all of it or also a debug subset
    max_samples_dev = config.MAX_SAMPLES_DEBUG if config.DEBUG_MODE and hasattr(config, 'MAX_SAMPLES_DEBUG_DEV') else None
    
    print("Preparing training dataset...")
    train_dataset = QuadrupletDataset(
        file_path=config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_source_length=config.MAX_SOURCE_LENGTH,
        max_target_length=config.MAX_TARGET_LENGTH,
        task_prefix=config.TASK_PREFIX,
        max_samples=max_samples,
        is_train=True
    )
    print("Preparing development (validation) dataset...")
    
    dev_dataset = QuadrupletDataset(
        file_path=config.DEV_FILE, # <--- 使用 dev.json
        tokenizer=tokenizer,
        max_source_length=config.MAX_SOURCE_LENGTH,
        max_target_length=config.MAX_TARGET_LENGTH,
        task_prefix=config.TASK_PREFIX,
        max_samples=max_samples_dev, # 可以为验证集也设置一个 DEBUG 数量
        is_train=True # 验证集也需要 'output' 字段进行评估
    )
    # For evaluation during training, you might want a validation set.
    # If you have one, load it similarly. If not, you can evaluate on a subset of train
    # or skip evaluation during training and do it post-hoc.
    # For simplicity, we'll skip validation set here, but it's highly recommended.
    # eval_dataset = QuadrupletDataset(...) 

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id # Important for Seq2Seq
    )
    # Custom compute_metrics for Trainer
    all_eval_results = [] # <--- 新增：用于收集所有轮次的评估结果

    def compute_metrics_for_trainer(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # --- 关键修改开始 ---
        # Replace -100 in predictions with the pad_token_id
        # This is crucial because model.generate() output (preds) might be padded by Trainer
        # with -100 if sequences have different lengths, and tokenizer.decode
        # doesn't expect -100 as a valid token_id for decoding.
        # It expects pad_token_id (e.g., 0 for T5) or actual vocab ids.
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        # --- 关键修改结束 ---

        # ---打印id---
        print("--- Raw Predictions (preds) ---")
        if isinstance(preds, np.ndarray) or isinstance(preds, torch.Tensor):
            print(preds.shape)
            # 打印前几个样本和一些统计信息
            for i in range(min(5, preds.shape[0])):
                print(f"Sample {i} (first 20 tokens): {preds[i, :20]}")
                print(f"Sample {i} min: {np.min(preds[i])}, max: {np.max(preds[i])}")
            print(f"Overall preds min: {np.min(preds)}, max: {np.max(preds)}")
        else:
            print(type(preds)) # 如果不是 tensor 或 ndarray，看看是什么
        
        # ... 然后是原来的解码代码

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Clean up decoded strings (T5 often adds <pad> or </s> at the beginning/end if not skipped properly)
        cleaned_preds = [pred.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").strip() for pred in decoded_preds]
        cleaned_labels = [label.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").strip() for label in decoded_labels]

        pred_quads_list = [parse_quadruplets(p) for p in cleaned_preds]
        gold_quads_list = [parse_quadruplets(l) for l in cleaned_labels]
        
        scores = compute_f1_scores(pred_quads_list, gold_quads_list, config.SIMILARITY_THRESHOLD)
        
        print(f"\n--- Evaluation Results (Epoch/Step specific) ---")
        for key, value in scores.items():
            print(f"{key}: {value:.4f}")
        print("--------------------------------------------------\n")
        
        # Store results for saving later (add epoch/step info if possible)
        # Trainer doesn't directly pass epoch to compute_metrics, but you can infer or log steps
        current_step = trainer.state.global_step # Get current step
        scores_to_log = scores.copy()
        scores_to_log["step"] = current_step
        all_eval_results.append(scores_to_log)

        # Return the metric to be used for early stopping and best model selection
        # Ensure this key matches `metric_for_best_model`
        return {"f1_avg": scores["f1_avg"], "f1_hard": scores["f1_hard"], "f1_soft": scores["f1_soft"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(config.OUTPUT_DIR, "training_checkpoints"),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE, # <--- 使用评估批次大小
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="epoch", # <--- 评估策略：每个epoch结束时评估
        save_strategy="epoch",       # <--- 保存策略：每个epoch结束时保存（如果更好）
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END, # <--- 训练结束后加载最佳模型
        metric_for_best_model="f1_avg", # <--- 选择最佳模型的指标 (需要与compute_metrics_for_trainer返回的key匹配)
        greater_is_better=True,         # <--- f1_avg 越大越好
        predict_with_generate=True,
        logging_dir=os.path.join(config.OUTPUT_DIR, "logs"),
        logging_steps=config.LOGGING_STEPS if hasattr(config, 'LOGGING_STEPS') else (100 if not config.DEBUG_MODE else 10),
        save_total_limit=2, # 只保存最好的和最新的一个（如果load_best_model_at_end=True）
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        generation_max_length=config.GENERATION_MAX_LENGTH,
        generation_num_beams=config.GENERATION_NUM_BEAMS,

        # eval_accumulation_steps=None, # 可以设置如果评估数据量大，显存不足
    )

    # Custom compute_metrics for Trainer (optional, for live eval during training)
    # This is a simplified version. The full F1 computation can be complex for Trainer's eval loop.
    # Often, it's easier to evaluate separately after training.
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset, # If using eval_dataset
        # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_trainer # If using eval_dataset and want live custom metrics
    )
    print("Starting training...")
    trainer.train() # resume_from_checkpoint 可以设为 True 或具体路径来续训

    print("Training finished.")
    # 保存所有轮次的评估结果到文件
    if all_eval_results:
        with open(config.EVAL_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_eval_results, f, indent=4)
        print(f"All evaluation results saved to {config.EVAL_RESULTS_FILE}")

    # 如果 load_best_model_at_end=True，trainer.model 现在就是最佳模型
    # 我们可以选择在这里保存它到一个更明确的最终路径，或者依赖 training_checkpoints 中最好的那个
    final_model_path = os.path.join(config.OUTPUT_DIR, "best_model_after_eval") # 或者 final_model
    if config.LOAD_BEST_MODEL_AT_END:
        print(f"Best model loaded. Saving it to {final_model_path}")
        trainer.save_model(final_model_path) # Trainer 会自动保存最佳模型到 checkpoint，这里是额外保存
        tokenizer.save_pretrained(final_model_path)
    else: # 如果没有加载最佳模型，就保存最后一个状态的模型
        final_model_path_last = os.path.join(config.OUTPUT_DIR, "last_model_state")
        print(f"Saving last model state to {final_model_path_last}")
        trainer.save_model(final_model_path_last)
        tokenizer.save_pretrained(final_model_path_last)

if __name__ == "__main__":
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    if not os.path.exists(os.path.join(config.OUTPUT_DIR, "training_checkpoints")):
        os.makedirs(os.path.join(config.OUTPUT_DIR, "training_checkpoints"))
    if not os.path.exists(os.path.join(config.OUTPUT_DIR, "logs")):
        os.makedirs(os.path.join(config.OUTPUT_DIR, "logs"))
        
    train_model()