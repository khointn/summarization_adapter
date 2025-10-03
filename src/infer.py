import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER_DIR = "/home/quang.pham/khointn/summarization_adapter_git/output_models/20251003_182009_qwen3-0.6b-lora-ctx512-drp0.05-r8/checkpoint-800"  # path to your saved LoRA
MAX_INPUT_TOKENS = 512
MAX_TARGET_TOKENS = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSTR_PREFIX = (
    "You are a helpful assistant that writes concise summaries of the given article.\n\n"
    "Task: Read the following article text and write a clear, accurate summary.\n\n"
    "Article:\n"
)
RESPONSE_PREFIX = "\n\nSummary:"

def build_prompt(doc: str) -> str:
    return f"{INSTR_PREFIX}{doc.strip()}{RESPONSE_PREFIX} "

def summarize(full_text: str) -> str:
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )

    model.to(DEVICE).eval()

    # Build prompt and tokenize (truncate input context)
    prompt = build_prompt(full_text)
    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_TOKENS,
            do_sample=False,          # deterministic
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    # Decode; keep only the part after "Summary:"
    full_out = tok.decode(out[0], skip_special_tokens=True)
    pretrained_summary = full_out.split(RESPONSE_PREFIX, 1)[-1].strip()

    # Attach LoRA adapter
    adapter_model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    try:
        adapter_model = model.merge_and_unload()
    except Exception:
        pass

    adapter_model.to(DEVICE)

    # Generate
    with torch.no_grad():
        out = adapter_model.generate(
            **enc,
            max_new_tokens=MAX_TARGET_TOKENS,
            do_sample=False,          # deterministic
            num_beams=1,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )

    # Decode; keep only the part after "Summary:"
    full_out = tok.decode(out[0], skip_special_tokens=True)
    trained_summary = full_out.split(RESPONSE_PREFIX, 1)[-1].strip()

    return pretrained_summary, trained_summary

FULL_TEXT = """
LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.
"""
pretrained_summary, trained_summary = summarize(FULL_TEXT)

print(pretrained_summary)
print("="*100)
print(trained_summary)
