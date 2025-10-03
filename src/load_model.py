import torch
import logging
from typing import Tuple, Literal, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model, PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(config: Dict, mode: Literal["train", "eval"] = "train") -> Tuple:
    logger.info("Load base model and tokenizer")
    
    if mode=="train":
        tokenizer = AutoTokenizer.from_pretrained(config["model"], 
                                                    use_fast=True, 
                                                    trust_remote_code=True, 
                                                    truncate=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        bnb_kwargs = {}
        dtype = torch.bfloat16 if torch.cuda.is_available() else None
        if config["load_4bit"]:
            bnb_kwargs = dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                device_map="auto",
            )

        base_model = AutoModelForCausalLM.from_pretrained(
            config["model"],
            torch_dtype=dtype if not config["load_4bit"] else None,
            trust_remote_code=True,
            **bnb_kwargs,
        )

        logger.info("Setup LoRA adapter")
        lora_cfg = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","W_pack"],
        )
        adapter_model = get_peft_model(base_model, lora_cfg)

        adapter_model.generation_config.update(
            max_new_tokens=config["max_target_tokens"],
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    elif mode=="eval":
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], 
                                                  use_fast=True, 
                                                  trust_remote_code=True, 
                                                  truncate=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # reload base model
        base_model = AutoModelForCausalLM.from_pretrained(config["base_model"], torch_dtype=torch.bfloat16)

        # attach LoRA weights from checkpoint
        adapter_model = PeftModel.from_pretrained(base_model, config["model"])

        # (optional) merge for faster inference
        try:
            adapter_model = adapter_model.merge_and_unload()
        except Exception:
            pass

        # put on device
        device = "cuda" # if torch.cuda.is_available() else "cpu"
        adapter_model.generation_config.update(
            max_new_tokens=config["max_target_tokens"],
            num_beams=1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        adapter_model = adapter_model.to(device)
        adapter_model.eval()

    return adapter_model, tokenizer