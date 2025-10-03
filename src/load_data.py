import os
import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset
from transformers import (
    PreTrainedTokenizerBase,
    set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompt template
INSTR_PREFIX = (
    "You are a helpful assistant that writes concise summaries of the given article.\n\n"
    "Task: Read the following article text and write a clear, accurate summary.\n\n"
    "Article:\n"
)
RESPONSE_PREFIX = "\n\nSummary:"

def format_example(ex: Dict[str, str], text_col: str, sum_col: str) -> Tuple[str, str]:
    src = (ex.get(text_col) or "").strip()
    tgt = (ex.get(sum_col) or "").strip()
    prompt = f"{INSTR_PREFIX}{src}{RESPONSE_PREFIX} "
    return prompt, tgt

def tokenize_and_mask(example: Dict[str, str], tokenizer, config) -> Dict[str, List[int]]:
    prompt, target = format_example(example, config["text_col"], config["summary_col"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][:config["max_input_tokens"]]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"][:config["max_target_tokens"]]

    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    input_ids = input_ids[:config["max_input_tokens"] + config["max_target_tokens"]]

    labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]
    labels = labels[:config["max_input_tokens"] + config["max_target_tokens"]]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

@dataclass
class DataCollatorForCausalLMWithMaskedLabels:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100
    pad_to_multiple_of: int = 8

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of and max_len % self.pad_to_multiple_of:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    
def load_data(tokenizer, config, is_eval=False) -> Tuple:
    logger.info("Load and tokenize dataset...")
    set_seed(config["seed"])

    if not os.path.exists(config["data_dir"]):
        ds_raw = load_dataset("cnn_dailymail", "1.0.0")
        ds_raw.save_to_disk(config["data_dir"])
    else:
        ds_raw = load_dataset("cnn_dailymail", "1.0.0", cache_dir=config["data_dir"])

    train_ds_raw, val_ds_raw, test_ds_raw = ds_raw["train"].select(range(50000)), ds_raw["validation"].select(range(5000)), ds_raw["test"]
    logger.info(f"Done loading. Train samples: {len(train_ds_raw)}, Val samples: {len(val_ds_raw)}")

    if not is_eval:
        train_ds = train_ds_raw.map(lambda ex: tokenize_and_mask(ex, tokenizer, config),
                                    remove_columns=train_ds_raw.column_names, desc="Tokenize train")
        val_ds = val_ds_raw.map(lambda ex: tokenize_and_mask(ex, tokenizer, config),
                                remove_columns=val_ds_raw.column_names, desc="Tokenize eval")
    else:
        train_ds = None
        val_ds = test_ds_raw.map(lambda ex: tokenize_and_mask(ex, tokenizer, config),
                        remove_columns=test_ds_raw.column_names, desc="Tokenize test")

    data_collator = DataCollatorForCausalLMWithMaskedLabels(tokenizer=tokenizer, pad_to_multiple_of=8)
    return train_ds, val_ds, data_collator
    
if __name__ == "__main__":
    load_data()