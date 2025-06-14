# dataset.py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from utils import load_json_data # Assuming utils.py is in the same directory

class QuadrupletDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_source_length, max_target_length, task_prefix, max_samples=None, is_train=True):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_prefix = task_prefix
        self.is_train = is_train # To know if 'output' field exists

        print(f"Loading data from {file_path}...")
        self.data = load_json_data(file_path, max_samples, is_train)
        
        self.inputs = []
        self.targets = [] if is_train else None

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        source_mask = self.inputs[index]["attention_mask"].squeeze()
        
        if self.is_train and self.targets:
            target_ids = self.targets[index]["input_ids"].squeeze()
            # For T5, labels should not ignore padding tokens before shifted by 1
            # The trainer's DataCollatorForSeq2Seq handles label shifting and padding correctly
            return {"input_ids": source_ids, "attention_mask": source_mask, "labels": target_ids}
        else:
            # For inference, we only need input_ids and attention_mask
             return {"input_ids": source_ids, "attention_mask": source_mask, "id": self.data[index]['id']}


    def _build(self):
        print("Tokenizing data...")
        for item in tqdm(self.data, desc="Tokenizing"):
            input_text = self.task_prefix + item['content']
            
            tokenized_input = self.tokenizer(
                input_text,
                max_length=self.max_source_length,
                padding="max_length", # Pad to max_length for consistent tensor sizes
                truncation=True,
                return_tensors="pt"
            )
            self.inputs.append(tokenized_input)

            if self.is_train and self.targets is not None:
                target_text = item['output']
                # For T5, labels are the target_ids. The model internally shifts them for decoder input.
                # The DataCollatorForSeq2Seq will handle padding for labels.
                # We need to set padding to 'max_length' for labels as well if we want to use Trainer's default eval
                # or if we are not using a data collator that handles this.
                # For DataCollatorForSeq2Seq, we can pass labels without padding, it will handle it.
                # However, to be safe and explicit for labels to be same length:
                tokenized_target = self.tokenizer(
                    target_text,
                    max_length=self.max_target_length,
                    padding="max_length", 
                    truncation=True,
                    return_tensors="pt"
                )
                self.targets.append(tokenized_target)