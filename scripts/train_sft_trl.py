import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "sft_outputs/qwen25coder7b")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))

TRAIN_FILES = [
    "RL/train_level1.parquet",
    "RL/train_level2.parquet",
    "RL/train_level3.parquet",
]

def load_parquets(files):
    datasets = [load_dataset("parquet", data_files=f, split="train") for f in files]
    ds = concatenate_datasets(datasets)

    if "response" in ds.column_names and "completion" not in ds.column_names:
        ds = ds.rename_column("response", "completion")

    if "prompt" not in ds.column_names or "completion" not in ds.column_names:
        raise ValueError(f"Dataset needs prompt + completion columns: {ds.column_names}")

    def prompt_to_text(prompt):
        if isinstance(prompt, list):
            parts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role:
                        parts.append(f"{role}: {content}")
                    else:
                        parts.append(str(content))
                else:
                    parts.append(str(msg))
            return "\n".join(parts)
        return str(prompt)

    def format_example(x):
        return {
            "text": prompt_to_text(x["prompt"]) + "\nassistant:\n" + str(x["completion"])
        }

    ds = ds.map(format_example)
    return ds

train_ds = load_parquets(TRAIN_FILES)

print("Train examples:", len(train_ds))

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    bf16=True,
    fp16=False,
    report_to=[],
)

trainer = SFTTrainer(
    model=MODEL_NAME,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved model to:", OUTPUT_DIR)