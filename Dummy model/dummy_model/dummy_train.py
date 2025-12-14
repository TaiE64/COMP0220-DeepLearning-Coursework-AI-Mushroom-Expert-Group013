import math
import os
import random
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer


# ======================
#  1. Config
# ======================

MODEL_SAVE_DIR = "checkpoints_dummy"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

BASE_MODEL_NAME = "distilgpt2"  # just for the tokenizer (weights NOT used)
MAX_SEQ_LEN = 64
BATCH_SIZE = 32
EPOCHS = 10          # keep <= 10 for dummy model
LR = 1e-3
TRAIN_FRACTION = 0.9
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# ======================
#  2. Dataset & Tokenisation
# ======================

class DialogueDataset(Dataset):
    def __init__(self, tokenized_texts: List[List[int]], max_len: int, pad_token_id: int):
        self.sequences = tokenized_texts
        self.max_len = max_len
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx]

        # truncate
        ids = ids[: self.max_len]

        # input_ids: [w1..w_{n-1}], labels: [w2..w_n]
        input_ids = ids[:-1]
        labels = ids[1:]

        # ensure at least length 2
        if len(input_ids) < 1:
            input_ids = [self.pad_token_id]
            labels = [self.pad_token_id]

        # pad to max_len-1 because we removed one token
        max_len_input = self.max_len - 1
        pad_len = max_len_input - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            labels = labels + [-100] * pad_len  # -100 for ignored index in loss

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_and_prepare_data(tokenizer, max_seq_len: int, train_fraction: float = 0.9):
    """
    Uses the DailyDialog dataset. We create simple sequences by joining turns in a dialog.
    """
    print("Loading dataset (daily_dialog)...")
    ds = load_dataset("roskoN/dailydialog", trust_remote_code=True)

    # We'll use only the 'train' split for dummy, and a subset to keep it small.
    dialogs = ds["train"]["utterances"]
    print("Total dialogues in train split:", len(dialogs))

    # Make it small on purpose for dummy model
    max_dialogs = 500  # you can adjust if you want slightly more/less
    dialogs = dialogs[:max_dialogs]

    joined_texts = []

    for dialog in dialogs:
        # dialog is a list of utterance strings
        utterances = [u.strip() for u in dialog if isinstance(u, str) and u.strip()]
        if not utterances:
            continue
        text = " <eot> ".join(utterances)  # end-of-turn marker
        joined_texts.append(text)
    
    print("Usable dialogs after cleaning:", len(joined_texts))

    print("Tokenizing texts...")
    tokenized = tokenizer(
        joined_texts,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        add_special_tokens=True,
    )

    all_input_ids = tokenized["input_ids"]

    # Shuffle & split
    random.shuffle(all_input_ids)
    split_idx = int(len(all_input_ids) * train_fraction)
    train_ids = all_input_ids[:split_idx]
    val_ids = all_input_ids[split_idx:]

    train_dataset = DialogueDataset(
        train_ids,
        max_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    val_dataset = DialogueDataset(
        val_ids,
        max_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_dataset, val_dataset


# ======================
#  3. Dummy LSTM Language Model
# ======================

class DummyLSTMChatbot(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 1, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, labels=None):
        # input_ids: (batch, seq_len)
        embed = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(embed)      # (batch, seq_len, hidden_dim)
        logits = self.fc(outputs)          # (batch, seq_len, vocab_size)

        loss = None
        if labels is not None:
            # Flatten for cross entropy: (batch * seq_len, vocab)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return logits, loss

    # @torch.no_grad()
    # def generate(self, input_ids, max_new_tokens: int = 30):
    #     """
    #     Very simple greedy generation.
    #     """
    #     self.eval()
    #     generated = input_ids.clone().to(next(self.parameters()).device)

    #     for _ in range(max_new_tokens):
    #         logits, _ = self.forward(generated)
    #         next_token_logits = logits[:, -1, :]  # last time step
    #         next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    #         generated = torch.cat([generated, next_token], dim=-1)

    #     return generated

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 30, temperature: float = 1.0, top_k: int = 50):
        self.eval()
        generated = input_ids.clone().to(next(self.parameters()).device)

        for _ in range(max_new_tokens):
            logits, _ = self.forward(generated)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            top_k_vals, top_k_indices = torch.topk(next_token_logits, k=min(top_k, next_token_logits.size(-1)), dim=-1)
            probs = torch.softmax(top_k_vals, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)

            generated = torch.cat([generated, next_token], dim=-1)

        return generated


# ======================
#  4. Training & Evaluation
# ======================

def compute_perplexity(loss: float) -> float:
    return math.exp(loss)


def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        _, loss = model(input_ids, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def eval_epoch(model, dataloader):
    model.eval()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        _, loss = model(input_ids, labels=labels)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def sample_dialogue(model, tokenizer, prompt: str):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(DEVICE)

    generated_ids = model.generate(input_ids, max_new_tokens=40)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    text = text.replace("<eot>", " ")  # turn into line breaks
    return text


def main():
    # Reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print("Loading tokenizer:", BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, val_dataset = load_and_prepare_data(
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        train_fraction=TRAIN_FRACTION,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    vocab_size = tokenizer.vocab_size
    print("Vocab size:", vocab_size)

    model = DummyLSTMChatbot(
        vocab_size=vocab_size,
        embed_dim=256,
        hidden_dim=512,
        num_layers=1,
        pad_token_id=tokenizer.pad_token_id,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss = eval_epoch(model, val_loader)

        train_ppl = compute_perplexity(train_loss)
        val_ppl = compute_perplexity(val_loss)

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f}, ppl: {train_ppl:.2f} | "
              f"Val loss: {val_loss:.4f}, ppl: {val_ppl:.2f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_SAVE_DIR, "dummy_lstm_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tokenizer_name": BASE_MODEL_NAME,
                "config": {
                    "vocab_size": vocab_size,
                    "embed_dim": 256,
                    "hidden_dim": 512,
                    "num_layers": 1,
                    "pad_token_id": tokenizer.pad_token_id,
                }
            }, save_path)
            print(f"→ Saved best model to {save_path}")

    # Generate a few sample dialogues
    prompts = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Can you explain mushrooms?",
    ]
    for p in prompts:
        print("\n=== Prompt ===")
        print(p)
        print("=== Dummy model response ===")
        print(sample_dialogue(model, tokenizer, p))


if __name__ == "__main__":
    main()
