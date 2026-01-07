import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    # BitsAndBytesConfig, 
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = os.path.join("Experiments", "data", "sft_train.jsonl")
OUTPUT_DIR = os.path.join("Experiments", "outputs", "tinyllama-ipb-sft-lora")
print(">>> SCRIPT STARTED <<<")
SYSTEM_PROMPT = (
    "You are an institutional assistant for the Polytechnic Institute of Bragança. "
    "Answer concisely and factually. If the information is not available, say you do not know."
)

# Dataset
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# QLoRA (4-bit)
"""bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)"""

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
)
model.config.use_cache = False

# LoRA config (TinyLlama/LLaMA-like modules)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora_config)

# Format messages to plain text (stable, independent of chat templates)
def format_example(example):
    msgs = example["messages"]
    # Ensure we always include the same system prompt (even if dataset already has it)
    # (If your jsonl already contains system, this keeps it consistent.)
    out = f"<|system|>\n{SYSTEM_PROMPT}\n"
    for m in msgs:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role == "user":
            out += f"<|user|>\n{content}\n"
        elif role == "assistant":
            out += f"<|assistant|>\n{content}\n"
    # Important: end with assistant tag so the model learns to answer
    if not out.rstrip().endswith("<|assistant|>"):
        out += "<|assistant|>\n"
    return out

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    fp16=True,
    optim="adamw_torch",
    report_to="none",
    max_length=512,          # <-- aqui é onde defines o comprimento
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    formatting_func=format_example,
    args=args,
)
print(">>> STARTING TRAINING <<<")
trainer.train()
print(">>> TRAINING FINISHED <<<")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Done. LoRA adapter saved to:", OUTPUT_DIR)
