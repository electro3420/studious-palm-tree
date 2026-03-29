# ========================================================
# ULTIMATE HYPER-DETAILED MAX-EFFICIENCY ZSC FOR LIMITED COMPUTE
# GPT2Tokenizer + 8 Real Datasets + Markov from actual data
# Streaming Chunked Loading + Deep Encoder (12 layers checkpointed) + Hybrid Fusion
# Dynamic Task Routing (QA / Summarization / Reasoning) + Robust Fallbacks
# Saves model to disk → inference from saved model only
# Designed to run within \~4-12 GB System RAM, low GPU, <1500 min
# ========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset as TorchDataset
import numpy as np
import re
import math
import time
import gc
import json
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any

# Allowed libraries
from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from transformers import GPT2Tokenizer

# ====================== DEVICE & SAFETY SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.empty_cache()
gc.collect()

MODEL_SAVE_PATH = "/content/hyper_zsc_final.pt"
START_TIME = time.time()
MAX_TIME_SECONDS = 1500 * 60  # 1500 minutes safety cap

# ====================== GPT2 TOKENIZER ======================
print("🔤 Loading GPT2Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ====================== DATASET LIST & SAFE LOADING ======================
DATASETS = [
    "Xerv-AI/ScienceLite",
    "Xerv-AI/GRAD",
    "Xerv-AI/Physics-dataset-700",
    "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "Jackrong/Qwen3.5-reasoning-700x",
    "allenai/ai2_arc",
    "nyu-mll/glue",
    "Rowan/hellaswag"
]

def load_dataset_safe(name: str, split="train", chunk_size=5000):
    try:
        print(f"Loading {name} ...")
        ds = load_dataset(name, split=split, streaming=False)
        # Chunk to pandas to control RAM
        chunks = []
        for i in range(0, len(ds), chunk_size):
            chunk = ds.select(range(i, min(i + chunk_size, len(ds)))).to_pandas()
            chunks.append(chunk)
            torch.cuda.empty_cache()
            gc.collect()
        df = pd.concat(chunks, ignore_index=True)
        print(f"   {name} loaded: {len(df)} rows")
        return df
    except Exception as e:
        print(f"   Failed to load {name}: {e}. Skipping with fallback.")
        return pd.DataFrame()  # empty fallback

# Load all datasets chunked
all_dfs = []
for ds_name in DATASETS:
    df_part = load_dataset_safe(ds_name)
    if not df_part.empty:
        all_dfs.append(df_part)
    if (time.time() - START_TIME) > MAX_TIME_SECONDS * 0.6:  # early stop if time tight
        break

df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
del all_dfs
torch.cuda.empty_cache()
gc.collect()

print(f"✅ Combined real datasets loaded safely: {len(df):,} rows total")

# ====================== DYNAMIC VOCAB ======================
def build_vocab_safe(df: pd.DataFrame, max_vocab=14000):
    print("🔤 Building vocab from combined real data in ultra-safe chunks...")
    counter = Counter()
    cols = ['text', 'question', 'answer', 'explanation', 'context', 'sentence', 'premise', 'hypothesis']
    for col in cols:
        if col in df.columns:
            for chunk in np.array_split(df[col].fillna(""), 50):
                text = " ".join(chunk.astype(str)).lower()
                tokens = re.findall(r'\b\w+\b', text)
                counter.update(tokens)
                torch.cuda.empty_cache()
                gc.collect()
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = idx
        idx += 1
    print(f"✅ Safe vocab built: {len(vocab)} tokens")
    return vocab

vocab = build_vocab_safe(df)

# ====================== DEEP ENCODER ======================
class DeepSafeEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=8, num_layers=12, max_len=192):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embed = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
                                                   dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pooler = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.GELU(), nn.Dropout(0.1), nn.Linear(embed_dim*2, embed_dim))
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.token_embed(x) + self.pos_embed[:, :seq_len, :]
        for layer in self.encoder.layers:
            x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False, src_key_padding_mask=mask)
        pooled = x.mean(dim=1)
        return F.normalize(self.pooler(pooled), p=2, dim=1)

# ====================== MARKOV FROM REAL DATA (CHUNKED) ======================
class SafeMarkovScorer:
    def __init__(self, order=2, smoothing=0.015):
        self.order = order
        self.smoothing = smoothing
        self.transitions = {}
        self.vocab = set(["<START>", "<END>", "<UNK>"])
    
    def build_from_real_data(self, df):
        print("🔨 Building Markov from combined real datasets (chunked)...")
        # Simple fallback grouping by available columns
        for idx, row in df.iterrows():
            if idx % 5000 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            texts = []
            for col in ['question', 'answer', 'explanation', 'text', 'context', 'sentence']:
                if col in df.columns and pd.notna(row.get(col)):
                    texts.append(str(row[col]))
            for text in texts:
                words = ["<START>"] + re.findall(r'\b\w+\b', text.lower()) + ["<END>"]
                self.vocab.update(words)
                for i in range(len(words) - self.order):
                    context = tuple(words[i:i+self.order])
                    next_w = words[i + self.order]
                    if "general" not in self.transitions:
                        self.transitions["general"] = defaultdict(lambda: defaultdict(float))
                    self.transitions["general"][context][next_w] += 1.0
        
        # Normalize
        if "general" in self.transitions:
            normalized = {}
            for context, counts in self.transitions["general"].items():
                total = sum(counts.values()) + self.smoothing * len(self.vocab)
                norm_dict = {w: (counts.get(w, 0.0) + self.smoothing) / total for w in list(counts.keys()) + list(self.vocab)}
                normalized[context] = norm_dict
            self.transitions["general"] = normalized
        print(f"✅ Markov built. Keys: {len(self.transitions)}")
    
    def score_text(self, text: str, label_key: str = "general") -> float:
        if label_key not in self.transitions:
            label_key = "general"
        words = ["<START>"] + re.findall(r'\b\w+\b', text.lower()) + ["<END>"]
        if len(words) <= self.order:
            return 0.0
        log_prob = 0.0
        trans = self.transitions.get(label_key, {})
        unk_prob = self.smoothing / (len(self.vocab) + self.smoothing)
        for i in range(len(words) - self.order):
            context = tuple(words[i:i + self.order])
            next_w = words[i + self.order]
            prob = trans.get(context, {}).get(next_w, unk_prob)
            log_prob += math.log(prob + 1e-12)
        return log_prob / max(1, len(words) - self.order)

# ====================== MAIN CLASSIFIER ======================
class HyperSafeZSC:
    def __init__(self):
        self.encoder = DeepSafeEncoder().to(device)
        self.markov_scorer = SafeMarkovScorer()
        self.optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=5e-5, weight_decay=1e-5)
        self.scaler = GradScaler()
        self.max_len = 192
        self.fusion_weights = {"semantic": 0.58, "markov": 0.42}
        self.task_router = {"qa": 1.15, "summarization": 0.95, "reasoning": 1.10, "generation": 1.05}
        self.grad_accum = 16
    
    def _text_to_ids_batch(self, texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        tensor = encoded['input_ids'].to(device)
        mask = (tensor == tokenizer.pad_token_id).to(device)
        return tensor, mask
    
    def train(self, df, epochs=3, batch_size=2):
        print("🚀 Starting training on combined real datasets...")
        self.markov_scorer.build_from_real_data(df)
        
        class SafeDS(TorchDataset):
            def __init__(self, df):
                self.df = df.reset_index(drop=True)
            def __len__(self): return len(self.df)
            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                q = str(row.get('question', row.get('text', '')))
                a = str(row.get('answer', row.get('label', '')))
                exp = str(row.get('explanation', row.get('context', '')))
                return q, a, exp, "general"
        
        loader = DataLoader(SafeDS(df), batch_size=batch_size, shuffle=True, pin_memory=True)
        
        self.encoder.train()
        for epoch in range(epochs):
            if time.time() - START_TIME > MAX_TIME_SECONDS * 0.8:
                print("⏰ Time limit approaching — stopping training early.")
                break
            total_loss = 0.0
            self.optimizer.zero_grad()
            for step, (qs, ans, exps, _) in enumerate(loader):
                with autocast(dtype=torch.float16):
                    q_ids, q_mask = self._text_to_ids_batch(qs)
                    a_ids, a_mask = self._text_to_ids_batch(ans)
                    e_ids, e_mask = self._text_to_ids_batch(exps)
                    q_emb = self.encoder(q_ids, q_mask)
                    a_emb = self.encoder(a_ids, a_mask)
                    e_emb = self.encoder(e_ids, e_mask)
                    sim = F.cosine_similarity(q_emb, a_emb) + 0.5 * F.cosine_similarity(q_emb, e_emb)
                    loss = -torch.log(sim.clamp(min=1e-7)).mean() / self.grad_accum
                self.scaler.scale(loss).backward()
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.8)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    gc.collect()
                total_loss += loss.item() * self.grad_accum
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f}")
            torch.cuda.empty_cache()
            gc.collect()
        self.encoder.eval()
        print("✅ Training finished within limits.\n")
    
    def save(self):
        torch.save({
            'encoder': self.encoder.state_dict(),
            'markov': self.markov_scorer.transitions,
            'markov_vocab': self.markov_scorer.vocab,
        }, MODEL_SAVE_PATH)
        print(f"✅ Model saved to {MODEL_SAVE_PATH}")
    
    def load(self):
        if os.path.exists(MODEL_SAVE_PATH):
            ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
            self.encoder.load_state_dict(ckpt['encoder'])
            self.markov_scorer.transitions = ckpt['markov']
            self.markov_scorer.vocab = ckpt.get('markov_vocab', self.markov_scorer.vocab)
            print(f"✅ Loaded saved model from {MODEL_SAVE_PATH}")
        else:
            print("No saved model — training first.")
    
    def classify(self, text: str, candidate_labels: List[str], task_type: str = "qa"):
        text_ids, text_mask = self._text_to_ids_batch([text])
        text_emb = self.encoder(text_ids, text_mask)
        lbl_ids, lbl_mask = self._text_to_ids_batch(candidate_labels)
        lbl_emb = self.encoder(lbl_ids, lbl_mask)
        sem = F.cosine_similarity(text_emb.repeat(len(candidate_labels), 1), lbl_emb, dim=1)
        sem_scores = dict(zip(candidate_labels, sem.tolist()))
        mark_scores = {lbl: self.markov_scorer.score_text(text, "general") for lbl in candidate_labels}
        final = {}
        boost = self.task_router.get(task_type, 1.0)
        for lbl in candidate_labels:
            s = sem_scores.get(lbl, 0.0)
            m = mark_scores.get(lbl, 0.0)
            m_norm = m / (max(mark_scores.values()) + 1e-8) if mark_scores else 0.0
            final[lbl] = self.fusion_weights["semantic"] * s * boost + self.fusion_weights["markov"] * m_norm
        pred = max(final, key=final.get)
        return {"predicted_label": pred, "score": round(final[pred], 4), "all_scores": {k: round(v, 4) for k, v in sorted(final.items(), key=lambda x: x[1], reverse=True)}}

# ====================== EXECUTION ======================
classifier = HyperSafeZSC()

if not os.path.exists(MODEL_SAVE_PATH):
    classifier.train(df, epochs=3, batch_size=2)
    classifier.save()
else:
    classifier.load()

# ====================== FINAL INFERENCE ======================
print("\n" + "="*120)
print("RUNNING FINAL INFERENCE FROM SAVED MODEL")
print("="*120)

test_cases = [
    {"text": "What causes a spark when removing synthetic clothes in dry weather?", "labels": ["static electricity", "electrostatics", "electric field"], "task": "qa"},
    {"text": "Summarize the principle of conservation of charge.", "labels": ["conservation of charge", "quantization", "electrostatics"], "task": "summarization"},
    {"text": "Explain step by step why like charges repel.", "labels": ["electrostatic force", "coulomb law", "electric field"], "task": "reasoning"}
]

for i, case in enumerate(test_cases, 1):
    print(f"\nTest Case {i} ({case['task']}):")
    print(f"Text: {case['text']}")
    result = classifier.classify(case['text'], case['labels'], task_type=case['task'])
    print(f"Predicted: {result['predicted_label']}")
    print(f"Score: {result['score']}")
    print("All Scores:", result['all_scores'])

print("\n✅ Full hyper-detailed training + inference completed within limits.")
print("Model saved and loaded from disk for all future inference.")
