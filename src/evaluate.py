import os
import gc
from argparse import ArgumentParser
import torch
import evaluate
import yaml
import logging

import wandb
from transformers import (
    TrainingArguments,
    Trainer,
)

from src.load_data import load_data
from src.load_model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(args) -> dict:
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    config["model"] = args.output_model
    config["base_model"] = args.pretrained_model

    run_name = f"eval_{config['model'].split('/')[-1]}"
    config["run_name"] = run_name
    config["output_dir"] = os.path.join(config["output_base_dir"], run_name)

    logger.info(config)

    return config

rouge_metric = evaluate.load("rouge")
bleu_metric  = evaluate.load("bleu")

@torch.no_grad()
def streaming_eval(
    model,
    tokenizer,
    val_ds,
    n_samples=256,
    batch_size=1,
    max_input_ctx=512,
    max_new_tokens=128,
    device=None,
    clear_cuda_every=8,         # free CUDA cache periodically
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    end = min(n_samples, len(val_ds))
    for start in range(0, end, batch_size):
        stop = min(start + batch_size, end)
        batch = val_ds.select(range(start, stop))

        # Truncate inputs
        input_ids = batch["input_ids"]
        attention = batch["attention_mask"]
        labels    = batch["labels"]

        if max_input_ctx is not None:
            input_ids = [ids[:max_input_ctx] for ids in input_ids]
            attention = [att[:max_input_ctx] for att in attention]

        # Padding
        max_len = max(len(x) for x in input_ids)
        pad_id = tokenizer.pad_token_id
        batch_input = [x + [pad_id]*(max_len - len(x)) for x in input_ids]
        batch_attn  = [x + [0]*(max_len - len(x))  for x in attention]

        batch_input = torch.tensor(batch_input, dtype=torch.long, device=device)
        batch_attn  = torch.tensor(batch_attn,  dtype=torch.long, device=device)

        # Generate and decode outputs
        gen = model.generate(
            input_ids=batch_input,
            attention_mask=batch_attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # Prepare and decode labels for this batch (replace -100 with pad for decode)
        max_lab = max(len(l) for l in labels)
        labels_padded = [
            [tok if tok != -100 else pad_id for tok in l] + [pad_id]*(max_lab - len(l))
            for l in labels
        ]
        decoded_refs = tokenizer.batch_decode(
            torch.tensor(labels_padded, dtype=torch.long), skip_special_tokens=True
        )

        # 6) Minimal post-processing for ROUGE-Lsum
        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_refs  = [r.strip() for r in decoded_refs]
        decoded_preds = ["\n".join(p.splitlines()) for p in decoded_preds]
        decoded_refs  = ["\n".join(r.splitlines()) for r in decoded_refs]

        # 7) Stream into metrics (no big lists kept)
        rouge_metric.add_batch(predictions=decoded_preds, references=decoded_refs)
        bleu_metric.add_batch(predictions=decoded_preds, references=[[r] for r in decoded_refs])

        # 8) Free everything we can
        del batch_input, batch_attn, gen
        if torch.cuda.is_available() and (start // batch_size) % clear_cuda_every == 0:
            torch.cuda.empty_cache()
        gc.collect()

    # 9) Compute final metrics
    rouge = rouge_metric.compute(use_stemmer=True)
    bleu  = bleu_metric.compute()
    return {
        "rouge1": round(rouge["rouge1"] * 100, 4),
        "rouge2": round(rouge["rouge2"] * 100, 4),
        "rougeL": round(rouge["rougeL"] * 100, 4),
        "rougeLsum": round(rouge["rougeLsum"] * 100, 4),
        "bleu": round(bleu["bleu"] * 100, 4),
    }

def eval_adapter(config) -> None:
    wandb.init(
    project=config["wandb_project"],
    name=config["run_name"],
    config=config)
        
    adapter_model, tokenizer = load_model(config, mode="eval")
    _, val_ds, _ = load_data(tokenizer, config)
    logger.info("Done setup model and data. Start evaluation.")

    scores = streaming_eval(
        adapter_model,
        tokenizer,
        val_ds,
        n_samples=100,
        batch_size=1,
        max_input_ctx=512,
        max_new_tokens=128,
    )
    print(scores)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_model", 
                        type=str, 
                        required=True, 
                        help="Path to the trained model")
    
    parser.add_argument("--pretrained_model", 
                        type=str, 
                        required=True,
                        help="Path to the trained model")
    args = parser.parse_args()

    config = load_config(args)
    eval_adapter(config)