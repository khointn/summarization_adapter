# SLMs Summarization Adapter

This repository provides code for **summarization adapter** built to work on top of language model (Qwen-3 0.6B, Qwen-3 1.7B).  The adapter provides concise summaries for long documents while avoiding the need to fine‑tune all of the base model's weights.  It leverages parameter‑efficient fine‑tuning (LoRA) to insert a small number of trainable parameters into a frozen base model. 

---

## 1. Adapter Architecture

### 1.1 Overview

The summarization adapter is built as a **LoRA (Low‑Rank Adaptation)** layer inserted into a frozen transformer model.  Instead of fine‑tuning all parameters of the base model, LoRA introduces a pair of low‑rank matrices that inject task‑specific information into each attention head.  This drastically reduces the number of trainable parameters (from billions to a few million) while retaining the capabilities of the underlying model.

At training time, only the LoRA parameters are updated; the base model’s weights remain unchanged.  At inference time, the adapter is merged with the base model to produce summaries with minimal overhead.

### 1.2 Integration with the Base Model

The adapter is integrated into the base model via [the **PEFT** library](https://github.com/huggingface/peft), which provides a unified interface for parameter‑efficient fine‑tuning on Hugging Face models.  The main steps are:

1. **Load the base model** (e.g. Qwen 2.5 0.6B) and its tokenizer from Hugging Face’s `transformers` library.
2. **Wrap the model with a LoRA configuration** using `peft.get_peft_model`.  The configuration specifies the rank `r` of the low‑rank matrices, the target modules (typically all attention layers), and other hyper‑parameters like `alpha` and `dropout`.
3. **Fine‑tune the adapter** on summarization data using causal or sequence‑to‑sequence loss.  Only the LoRA parameters are updated.
4. **Save the adapter** separately from the base model.  During inference, you can load the base model and merge the LoRA weights into it using `model = PeftModel.from_pretrained(base_model, adapter_path, merge=True)`.

<!-- ### 1.3 Key Components

| Component            | Description                                                                                                                                                                                                                                        |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Base Model**       | A pretrained transformer used as the backbone.  The default demonstration uses `google/flan‑t5‑small` to minimise resources, but you can swap it with `Qwen/QWen‑VL‑Chat‑3B`, `meta‑llama/Llama‑3‑3b‑chat`, or `mistralai/Mistral‑7B‑Instruct` when training on a machine with a GPU. |
| **Tokenizer**        | Converts raw text into tokens.  The same tokenizer as the base model must be used for both training and inference.                                                                                                                                |
| **LoRA Adapter**     | A set of small low‑rank matrices inserted into the attention projections of the base model.  Each LoRA layer has rank `r` (e.g. 8 or 16) and scales updates by a factor `alpha/r`.                                                              |
| **Training Loop**    | Implements supervised fine‑tuning on summarization data.  It handles batching, tokenisation, padding, forward/backward passes and optimiser steps.  Only the LoRA parameters require gradients.            |
| **Evaluation Module**| Computes metrics such as ROUGE‑1/2/L, BLEU and BERTScore on a validation set.  It can compare the adapter’s outputs with those of the base model to quantify improvements.                                |
| **Data Preprocessor**| Reads raw documents, optionally truncates/segments long documents, and converts them into the expected JSON format.  It can also split the dataset into train/validation/test splits.                        |

### 1.4 Rationale and Advantages

* **Parameter Efficiency:**  Fine‑tuning the full 3B base model is expensive.  LoRA reduces the number of trainable parameters by orders of magnitude, which makes it feasible to run on consumer GPUs or even CPU with small models.
* **Modularity:**  The adapter is saved separately and can be applied to multiple base models.  You can train different adapters for different domains without duplicating the entire model.
* **Compatibility:**  Because the base model is frozen, you preserve its capabilities and mitigate catastrophic forgetting.  The same training script can be reused across QWen, Llama, Mistral or other architectures supported by `transformers`.

--- -->

## 2. Data and Formats

The model accepts **plain text documents** as input. This works use the **dailymail** dataset [1] and the **billsum** dataset [2].

[1]: https://huggingface.co/datasets/abisee/cnn_dailymail "CNN Dailymail · Datasets at Hugging Face"
[2]: https://huggingface.co/datasets/FiscalNote/billsum "FiscalNote/billsum · Datasets at Hugging Face"


```json
{
  "article": "Full legal documentation text",
  "highlights": "Summary of the text"
}
```

When training, documents longer than the model's maximum sequence length will be **chunked** into overlapping segments.

The adapter produces a ```summary``` as plain text from the given ```article```. 

## 3. Evaluation Metrics

| Metric            | Purpose and Usage                                                                                                                                                                                                     |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ROUGE‑1/2/L**   | Measures overlap of unigrams, bigrams and longest common subsequence between the generated summary and the reference summary.  Higher ROUGE indicates better coverage of important information.                      |
| **BLEU**          | Traditionally used in machine translation, BLEU measures the precision of n‑grams in the generated summary.  We use BLEU‑4 for completeness.                                                                        |
<!-- | **BERTScore**     | Computes similarity in embedding space using a pretrained BERT model.  It captures semantic similarity beyond exact n‑gram overlap.                                                                                  |
| **Length Ratio**  | Average ratio of generated summary length to reference length.  Helps ensure summaries are concise without being too short.                                                                                          |
| **Readability**   | Optional metric based on readability formulas (e.g. Flesch Reading Ease) to ensure summaries are easy to read.                                                                                                      | -->

<!-- During evaluation, we compare the adapter’s metrics against those of the base model.  Improvements in ROUGE and BERTScore indicate that the adapter is effectively capturing salient information.  Length ratio and readability help tune hyper‑parameters such as maximum summary length and `temperature` during generation. -->

## 4. How to run

Setup venv and install dependencies:
```bash
conda create -n adapter_venv python=3.11
conda activate adapter_venv
(adapter_venv) pip install -r requirements.txt
```

For training:
```bash
(adapter_venv) python -m src.train --pretrained_model <PRETRAINED_MODEL>
```

For evaluation:
```bash
(adapter_venv) python -m src.evaluate --pretrained_model <PRETRAINED_MODEL> --output_model <OUTPUT_MODEL_PATH>
```

For inference:
```bash
(adapter_venv) python -m src.infer
```
<!-- ## 6. Challenges and Solutions

| Challenge                        | Mitigation                                                                                                                                                                                         |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Long Documents**              | Transformer models have a fixed context length (e.g. 4k tokens).  Long documents must be segmented.  Use the preprocessing script to split documents into overlapping windows, generate segment summaries, then recursively summarise. |
| **Memory Constraints**          | Large base models (3B parameters) require significant GPU memory.  Use quantisation (8‑bit or 4‑bit with `bitsandbytes`), gradient checkpointing and low‑rank adaptation.  Alternatively, train on a smaller model for prototyping.     |
| **Hallucination**               | Base models may introduce factual errors.  Consider using factual consistency metrics (e.g. QAG or FactCC) and penalise hallucination via reinforcement learning or post‑editing.                                                  |
| **Evaluation Metric Limitations**| ROUGE and BLEU may not fully capture semantic correctness.  Use BERTScore or human evaluation when possible.  Provide multiple metrics to get a comprehensive view of performance.                                              |
| **Domain Mismatch**             | If training on news data and deploying on scientific articles, summaries may not generalise.  To mitigate, train separate adapters per domain or use multi‑domain training.                                                         | -->

---