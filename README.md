# 📝 Full DHMM-GPT Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================
# Differentiable Hidden Markov Module
# =========================================================
class DifferentiableHMM(nn.Module):
    def __init__(self, num_states, embed_size):
        super().__init__()
        self.num_states = num_states
        self.embed_size = embed_size

        self.transitions = nn.Parameter(torch.randn(num_states, num_states))
        self.emitter = nn.Linear(num_states, embed_size)

    def forward(self, state_dist):
        # transition
        trans_probs = torch.softmax(self.transitions, dim=-1)
        next_state_dist = torch.matmul(state_dist, trans_probs)
        # emission
        emission = self.emitter(next_state_dist)
        return next_state_dist, emission


# =========================================================
# Multi-Head Self Attention
# =========================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, E = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot product
        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).transpose(1, 2).contiguous().view(B, L, E)

        return self.fc_out(out)


# =========================================================
# Transformer Block with HMM
# =========================================================
class DHMMBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden, num_states, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

        self.hmm = DifferentiableHMM(num_states, embed_size)
        self.num_states = num_states

    def forward(self, x, state_dist, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        # evolve HMM
        next_state_dist, emission = self.hmm(state_dist)
        x = x + emission.unsqueeze(1)

        return x, next_state_dist


# =========================================================
# DHMM-GPT Model
# =========================================================
class DHMMGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=6,
                 num_heads=8, ff_hidden=1024, num_states=16, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            DHMMBlock(embed_size, num_heads, ff_hidden, num_states, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self.num_states = num_states

    def forward(self, input_ids, mask=None, labels=None):
        B, L = input_ids.shape
        tokens = self.token_emb(input_ids)
        positions = self.pos_emb(torch.arange(L, device=input_ids.device)).unsqueeze(0).expand(B, -1, -1)
        x = tokens + positions

        # initial uniform HMM state
        state_dist = torch.ones(B, self.num_states, device=input_ids.device) / self.num_states

        for layer in self.layers:
            x, state_dist = layer(x, state_dist, mask)

        x = self.norm(x)
        logits = self.fc_out(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))

        return {"logits": logits, "loss": loss}
```

---

# 🔎 Explanation

### 1. **Differentiable HMM**

* Each block evolves a **latent state distribution** using a softmaxed transition matrix.
* The distribution emits embeddings added into the hidden stream.

### 2. **Attention**

* Standard GPT-2 causal attention.
* Masking ensures autoregressive behavior.

### 3. **Feedforward + Residual**

* Two-layer MLP with ReLU.
* Residual + LayerNorm like GPT-2.

### 4. **Model Output**

* Vocabulary logits for causal language modeling.
* Optional **cross-entropy loss** if labels are passed.

---

⚡ This is a **baseline DHMM-GPT**.
👉 Do you want me to extend this version with **internal noise mutations + dithering filters** (like in AEON-MACHINE), or keep this “pure DHMM-GPT” first?
