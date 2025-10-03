import os
from argparse import ArgumentParser
from datetime import datetime
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

    print(config)
    if args.pretrained_model is not None:
        config["model"] = args.pretrained_model

    model_name = config["model"].split("/")[-1].lower()
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}-lora-ctx{config['max_input_tokens']}-drp{config['lora_dropout']}-r{config['lora_r']}"
    
    config["model_name"] = model_name
    config["run_name"] = run_name
    config["output_dir"] = os.path.join(config["output_base_dir"], run_name)

    return config

def train_adapter(config) -> None:
    wandb.init(
    project=config["wandb_project"],
    name=config["run_name"],
    config=config)
        
    adapter_model, tokenizer = load_model(config)
    train_ds, val_ds, data_collator = load_data(tokenizer, config)
    logger.info("Done setup model and data. Start training.")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=config["warmup_ratio"],
        logging_steps=config["logging_steps"],
        save_strategy="steps",
        save_steps=config["eval_steps"],
        eval_strategy="steps",
        # eval_steps=config["eval_steps"],

        fp16=False,
        load_best_model_at_end=True,
        # metric_for_best_model="rougeL",
        greater_is_better=True,
        seed=config["seed"],
        optim=config["optim"],

        report_to="wandb",
        run_name=config["run_name"],
    )


    trainer = Trainer(
        model=adapter_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save LoRA adapter after training
    logger.info("Saving LoRA adapter and tokenizer.")
    adapter_dir = os.path.join(config["output_dir"], "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(config["output_dir"])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model", 
                        type=str, 
                        required=False, 
                        default=None, 
                        help="Pretrained model. Default uses config.yaml")
    args = parser.parse_args()

    config = load_config(args)
    train_adapter(config)