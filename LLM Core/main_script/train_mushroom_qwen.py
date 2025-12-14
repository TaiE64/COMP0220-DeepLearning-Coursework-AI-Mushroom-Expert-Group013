import os
import torch
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# 🟢 CUDA memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ======================
# Basic configuration
# ======================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATA_FILE = "data/final_training_data.jsonl"
OUTPUT_DIR = "./qwen25_mushroom_qlora"

def load_jsonl_data(filepath):
    """Load JSONL formatted data"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(item)
    return data

def main():
    # 1. Load data
    print(f"Loading mushroom Q&A data from {DATA_FILE} ...")
    all_data = load_jsonl_data(DATA_FILE)
    print(f"✅ Loaded {len(all_data)} Q&A pairs")

    # Split into train and validation sets (90% train, 10% validation)
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    print(f"📊 Train: {len(train_data)} samples, Eval: {len(eval_data)} samples")

    # Convert to HuggingFace Datasets
    train_ds = Dataset.from_dict({"text": [item["text"] for item in train_data]})
    eval_ds = Dataset.from_dict({"text": [item["text"] for item in eval_data]})

    # Preview data
    print("\n" + "=" * 80)
    print("📝 Sample training data:")
    print("=" * 80)
    for i in range(min(3, len(train_data))):
        print(f"\n[Sample {i+1}]")
        print(train_data[i]["text"][:200] + "...")
    print("=" * 80)

    # 2. Configure 4-bit quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load tokenizer
    print(f"\n🔧 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Load model (4-bit quantization)
    print(f"🔧 Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 5. Configure LoRA
    lora_config = LoraConfig(
        r=16,                          # LoRA rank
        lora_alpha=32,                 # LoRA alpha (commonly set to r*2)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Print trainable parameter statistics
    model.print_trainable_parameters()

    # 6. Formatting function: directly use the text field
    def formatting_func(example):
        # Data format is already "User: xxx\nAssistant: xxx<|im_end|>"
        return example["text"] + tokenizer.eos_token

    # 7. Training parameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     # Mushroom dataset is small; train more epochs

        # Batch size settings
        per_device_train_batch_size=4,          # Adjust based on data size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,          # Effective batch size = 4*8 = 32

        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing fork warnings
    )

    # 8. Use SFTTrainer for supervised fine-tuning
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    # 9. Start training
    print("\n" + "=" * 80)
    print("🚀 Starting QLoRA fine-tuning (Mushroom Knowledge Dataset)")
    print("=" * 80)
    trainer.train()

    # 10. Save LoRA weights
    print("\n💾 Saving LoRA adapter...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"✅ Training complete! Model saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
