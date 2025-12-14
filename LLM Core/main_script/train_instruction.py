import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
import torch
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 🟢 CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"   

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 1) Load tokenizer + model from your core model (Qwen 2.5)
# Make sure BASE_MODEL_PATH points to your Qwen model directory or HuggingFace ID
print(f"Loading tokenizer: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# Configure 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

# 2) Load Dolly 15K instruction dataset
print("Loading Dolly 15K...")
dolly = load_dataset("databricks/databricks-dolly-15k", split="train")

# Dolly columns: instruction, context, response, category :contentReference[oaicite:2]{index=2}

def format_dolly(example):
    instr = example["instruction"].strip()
    resp = example["response"].strip()

    # Single-turn QA style, no multi-turn
    # IMPORTANT: Add eos_token so the model learns to stop generating!
    text = f"User: {instr}\nBot: {resp}{tokenizer.eos_token}"
    return {"text": text}

dolly_formatted = dolly.map(format_dolly, remove_columns=dolly.column_names)

# Optional: sample a subset for speed (e.g. 5000 examples)
dolly_formatted = dolly_formatted.shuffle(seed=42).select(range(5000))
print("Dolly examples used:", len(dolly_formatted))

# 3) Tokenize
def tokenize_fn(batch):
    tokens = tokenizer(
        batch["text"],
        max_length=256,
        truncation=True,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dolly_formatted.map(tokenize_fn, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 4) Training args for stage 2
training_args = TrainingArguments(
    output_dir="core_model_instruction",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,          # short extra fine-tune
    learning_rate=5e-5,
    warmup_steps=100,
    logging_steps=50,
    save_strategy="epoch",

    fp16=False,
    bf16=True,             # Use BF16 for Qwen 2.5 if verified supported (Ampere+), else fp16
    gradient_checkpointing=True, # Critical for memory saving
    optim="paged_adamw_8bit",    # Use 8-bit optimizer to save memory
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("Starting Stage 2 instruction fine-tune...")
trainer.train()

print("Saving instruction-tuned model...")
trainer.save_model("core_model_instruction")
tokenizer.save_pretrained("core_model_instruction")
print("Done.")
