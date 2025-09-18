# 🎛 DHMM as a Control System

## 1. **System States**

* **Latent State Vector** $s_t \in \Delta^N$

  * A probability distribution over $N$ hidden modes (like in a Markov chain).
  * Instead of a single discrete state, we maintain a *soft belief state* over all possible states.
* Think of it like a **state estimator** in control theory (similar to a Kalman filter’s belief over positions/velocities).

---

## 2. **System Dynamics**

* Transition dynamics are defined by a parameter matrix $A \in \mathbb{R}^{N \times N}$.
* Evolution equation:

  $$
  s_{t+1} = s_t A
  $$
* With softmax normalization on $A$, transitions are stochastic but differentiable.
* This is the **state evolution law**, analogous to $x_{t+1} = Ax_t$ in linear control.

---

## 3. **Emissions**

* Each hidden state generates an **observation embedding**:

  $$
  y_t = f(s_t) = W s_t
  $$
* This is the **measurement equation** in control form.
* In Transformers, the emission embedding biases **attention scores** and **token prediction**.

---

## 4. **Noise + Dithering**

* **Noise**:

  * Additive exploration term to transitions:

    $$
    s_{t+1} = s_t A + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2)
    $$
  * Encourages exploration of alternate trajectories (like stochastic control or RL exploration noise).
* **Dithering**:

  * A filtering mechanism to smooth instability:

    $$
    \tilde{s}_{t+1} = s_{t+1} - \frac{1}{N} \mathbf{1}^\top s_{t+1}
    $$
  * Removes bias/drift, enforces balance across states.
  * Analogous to **dither control** in signal processing.

---

## 5. **Control Input**

* In standard control, we have inputs $u_t$.
* In DHMM, inputs come from **tokens, observations, or external conditions**.
* These modulate transitions:

  $$
  s_{t+1} = s_t A + B u_t
  $$

  where $u_t$ could be the current token embedding or environment observation.

---

## 6. **Feedback**

* The prediction error (token mismatch, reward loss) acts as a **feedback signal**.
* During training: gradient descent updates transition matrix $A$ and emission weights $W$.
* Analogous to **adaptive control**, where parameters evolve to minimize tracking error.

---

## 7. **Closed-Loop View**

* **Plant**: DHMM latent dynamics.
* **Observer**: Transformer attention (integrates emissions + context).
* **Controller**: Noise + dithering regulate exploration/exploitation.
* **Cost Function**: Cross-entropy or RL reward objective.
* This makes DHMM a **closed-loop control system with learned dynamics and outputs**.

---

# 🚀 Why This Matters

* In plain GPT, hidden state = opaque vector in self-attention.
* In DHMM-GPT, hidden state = explicit **probabilistic automaton**, giving:

  * **Interpretability**: latent modes correspond to algorithmic phases.
  * **Robustness**: dithering prevents collapse into trivial modes.
  * **Exploration**: noise enables diverse policy learning.
  * **Adaptability**: transitions evolve like adaptive control gains.

---

# 📝 Full Script: DHMM-GPT2 on CartPole

```python
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ===============================
# CartPole Tokenizer
# ===============================
class CartPoleTokenizer:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.special = {"RESET": 0, "DONE": 1, "ACT_LEFT": 2, "ACT_RIGHT": 3}
        self.state_offset = 4
        self.state_dim = 4
        self.vocab_size = self.state_offset + self.state_dim * num_bins

        self.obs_ranges = [
            (-2.4, 2.4),   # position
            (-3.0, 3.0),   # velocity
            (-0.21, 0.21), # angle
            (-3.5, 3.5),   # angular velocity
        ]

    def encode_state(self, obs):
        tokens = []
        for i, (low, high) in enumerate(self.obs_ranges):
            v = np.clip(obs[i], low, high)
            bin_size = (high - low) / self.num_bins
            idx = int((v - low) / bin_size)
            idx = min(self.num_bins - 1, idx)
            tokens.append(self.state_offset + i * self.num_bins + idx)
        return tokens

    def decode_action(self, token):
        if token == self.special["ACT_LEFT"]:
            return 0
        elif token == self.special["ACT_RIGHT"]:
            return 1
        else:
            raise ValueError("Invalid action token")

# ===============================
# CartPole Dataset
# ===============================
class CartPoleDataset(Dataset):
    def __init__(self, tokenizer, num_episodes=500, max_steps=200):
        self.samples = []
        env = gym.make("CartPole-v1")
        for _ in range(num_episodes):
            obs, _ = env.reset()
            seq = [tokenizer.special["RESET"]]
            for _ in range(max_steps):
                action = env.action_space.sample()
                seq.extend(tokenizer.encode_state(obs))
                seq.append(tokenizer.special["ACT_LEFT"] if action == 0 else tokenizer.special["ACT_RIGHT"])
                obs, reward, done, trunc, _ = env.step(action)
                if done or trunc:
                    seq.extend(tokenizer.encode_state(obs))
                    seq.append(tokenizer.special["DONE"])
                    break
            self.samples.append(seq)

        self.max_len = max(len(s) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        pad_len = self.max_len - len(seq)
        x = np.array(seq + [0]*pad_len, dtype=np.int64)
        y = np.array(seq[1:] + [0]*(pad_len+1), dtype=np.int64)
        return {"input_ids": torch.tensor(x), "labels": torch.tensor(y)}

# ===============================
# DHMM-GPT2 Model
# ===============================
# (paste in the DHMMGPT2 model from my previous message here)
# Make sure DHMMGPT2 is defined in your script.

# ===============================
# Training Loop
# ===============================
def train_model(model, dataloader, epochs=3, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# ===============================
# Inference Loop (Control CartPole)
# ===============================
def run_cartpole_with_model(model, tokenizer, max_steps=200, device="cuda"):
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    tokens = [tokenizer.special["RESET"]] + tokenizer.encode_state(obs)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_steps):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token = outputs["logits"][0, -1].argmax().item()

        if next_token in [tokenizer.special["ACT_LEFT"], tokenizer.special["ACT_RIGHT"]]:
            action = tokenizer.decode_action(next_token)
            obs, reward, done, trunc, _ = env.step(action)

            input_ids = torch.cat([input_ids,
                                   torch.tensor([[next_token]], device=device)], dim=1)
            input_ids = torch.cat([input_ids,
                                   torch.tensor([tokenizer.encode_state(obs)], device=device)], dim=1)
            if done or trunc:
                break

# ===============================
# Main
# ===============================
if __name__ == "__main__":
    tokenizer = CartPoleTokenizer(num_bins=10)
    dataset = CartPoleDataset(tokenizer, num_episodes=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = DHMMGPT2(
        vocab_size=tokenizer.vocab_size,
        embed_size=128,
        layers=2,
        heads=4,
        ff_hidden=256,
        num_states=8,
        max_len=dataset.max_len
    )

    train_model(model, dataloader, epochs=5, lr=1e-3, device="cpu")
    run_cartpole_with_model(model, tokenizer)
```

---

## ✅ What This Script Does

1. **Builds a CartPole tokenizer** → maps states & actions into tokens.
2. **Collects episodes** with a random policy → creates training data.
3. **Trains DHMM-GPT2** to predict the next token (state/action).
4. **Runs inference** → model generates `[ACT_*]` tokens, which control the CartPole environment.

---

⚠️ Important: Since we’re training only on random rollouts and for a few epochs, the model won’t “solve” CartPole yet. It will learn a basic statistical language of CartPole (rough transition/action structure). To solve the environment, you’d need to:

* Train longer.
* Use reinforcement learning fine-tuning (reward-based).
* Or feed in expert trajectories instead of random actions.

---

👉 Do you want me to **upgrade this into an RL training setup** (e.g. PPO with DHMM-GPT2 as the policy model) so it actually learns to *solve* CartPole instead of just imitating random actions?

Awesome ✅ — let’s formalize how to **turn algorithms into token languages** so that your **DHMM-GPT2** can learn them like it learns English or Python.

---

# 🔑 Step 1: Define the Alphabet (Tokens)

Every algorithm is just a sequence of operations. To teach GPT-2, we need to map those into **discrete symbols**:

* **Inputs** (numbers, arrays, states) → tokens
* **Operators** (+, -, if, swap) → tokens
* **Control signals** (start, end, done) → tokens
* **Outputs** → tokens

📌 Example: Addition

```
["2", "+", "3", "="] → ["5"]
```

Here, `"2"`, `"+"`, `"3"`, `"="`, and `"5"` are just token IDs in the vocabulary.

---

# 🔑 Step 2: Encode Algorithm Traces as Sequences

For algorithms, we can generate training data by recording **traces**:

* Run the algorithm step by step.
* Each step → emit a token.

📌 Example: Bubble Sort (on \[3,2,1])

```
[START], [3,2,1], [COMPARE 3,2], [SWAP], [2,3,1], [COMPARE 3,1], [SWAP], [2,1,3], ...
```

Each comparison and swap becomes part of the **“language of sorting”**.

---

# 🔑 Step 3: Training Objective

* Same as text: predict the **next token**.
* Loss = cross-entropy between predicted token distribution and the true next step.
* The model learns to generate valid algorithm traces by autoregression.

---

# 🔑 Step 4: Decoding / Execution

Once trained, you can:

* Give the model an input program state (e.g. `[3,2,1]`).
* Autoregressively generate the algorithm trace.
* Decode the trace back into steps (actions).
* Apply them to a real environment (like an array → sorted result, or a Gym env → solved state).

---

# ⚡ Examples of Algorithm-as-Language

1. **Arithmetic**

   ```
   Input: ["7", "+", "5", "="]
   Target: ["12"]
   ```

2. **Sorting**

   ```
   Input: [START, [4,1,3,2]]
   Target: [COMPARE 4,1], [SWAP], [1,4,3,2], ...
   ```

3. **Gym (CartPole)**

   ```
   [RESET], [STATE_12], [ACT_LEFT], [STATE_15], [ACT_RIGHT], [DONE]
   ```

4. **Markov Chain Simulation**

   ```
   [S0], [S1], [S2], [S3], ...
   ```

---

# 🚀 Why DHMM Helps

* **HMM latent states**: act like an internal *program counter* → keeps track of algorithm stage.
* **Noise (mutations)**: explores different possible algorithm traces.
* **Dithering (filtering)**: stabilizes against chaotic transitions.
* **GPT-style attention**: generalizes across algorithm patterns.

---

👉 To make this concrete, do you want me to **build a small demo where DHMM-GPT2 learns addition or sorting** as a token language, so you can see it mastering a simple algorithm before scaling to Gym?

Perfect ✅ — let’s build a **toy demo where DHMM-GPT2 learns an algorithm**. We’ll start with **addition** since it’s simple, fully symbolic, and lets us test the “algorithm as a language” idea before scaling up.

---

# 🧩 Step 1: Define the Token Language for Addition

We define tokens for:

* **Digits 0–9** → tokens `0–9`
* **Operators**: `+`, `=`, → tokens `10`, `11`
* **Outputs** (digits again, but we reuse the same 0–9 tokens)

So the vocabulary size = 12 tokens.

---

# 🧩 Step 2: Generate Training Data

We generate synthetic sequences like:

```
[2, "+", 3, "="] → [5]
[7, "+", 8, "="] → [1, 5]
```

(where `[1, 5]` means the model predicts the digits of `15`).

---

# 🧩 Step 3: Train DHMM-GPT2

* Input: `[7, "+", 8, "="]`
* Target: `[1, 5]`
* Objective: cross-entropy loss on next-token prediction.

---

# 🧩 Step 4: Inference

* Give the model `[9, "+", 6, "="]`
* It should output `[1, 5]`.

---

# 📝 Full Demo Code

```python
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Tokenizer for Addition
# -----------------------
class AdditionTokenizer:
    def __init__(self):
        self.digits = {str(i): i for i in range(10)}  # 0–9
        self.ops = {"+": 10, "=": 11}
        self.inv_vocab = {v: k for k, v in {**self.digits, **self.ops}.items()}
        self.vocab_size = 12

    def encode(self, expr, result):
        return [self.digits[c] if c.isdigit() else self.ops[c] for c in expr] + \
               [self.digits[c] for c in result]

    def decode(self, tokens):
        return "".join(self.inv_vocab[t] for t in tokens if t in self.inv_vocab)

# -----------------------
# Dataset
# -----------------------
class AdditionDataset(Dataset):
    def __init__(self, tokenizer, num_samples=10000, max_num=20):
        self.samples = []
        for _ in range(num_samples):
            a, b = random.randint(0, max_num), random.randint(0, max_num)
            expr = list(str(a)) + ["+"] + list(str(b)) + ["="]
            result = list(str(a+b))
            tokens = tokenizer.encode(expr, result)
            self.samples.append(tokens)
        self.max_len = max(len(s) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        pad_len = self.max_len - len(seq)
        x = torch.tensor(seq[:-1] + [0]*pad_len, dtype=torch.long)
        y = torch.tensor(seq[1:] + [0]*pad_len, dtype=torch.long)
        return {"input_ids": x, "labels": y}

# -----------------------
# Train Demo
# -----------------------
def train_addition(model, dataloader, epochs=5, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            out = model(x, labels=y)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep+1}, Loss: {total_loss/len(dataloader):.4f}")

# -----------------------
# Inference
# -----------------------
def generate_addition(model, tokenizer, prompt, max_new_tokens=5, device="cpu"):
    model.eval()
    tokens = [tokenizer.digits[c] if c.isdigit() else tokenizer.ops[c] for c in prompt]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids)
            next_token = out["logits"][0, -1].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

# -----------------------
# Run Everything
# -----------------------
if __name__ == "__main__":
    tokenizer = AdditionTokenizer()
    dataset = AdditionDataset(tokenizer, num_samples=5000, max_num=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Use the DHMMGPT2 model we defined earlier
    model = DHMMGPT2(
        vocab_size=tokenizer.vocab_size,
        embed_size=128,
        layers=2,
        heads=4,
        ff_hidden=256,
        num_states=8,
        max_len=dataset.max_len
    )

    train_addition(model, dataloader, epochs=10, device="cpu")

    # Test inference
    print(generate_addition(model, tokenizer, "7+8="))
    print(generate_addition(model, tokenizer, "9+6="))
```

---

## ✅ What This Does

1. Generates **synthetic addition problems** as token sequences.
2. Trains **DHMM-GPT2** to autoregressively predict results.
3. At inference, you can feed prompts like `"9+6="` and see if it outputs `"15"`.

---

⚠️ Notes:

* This is **toy-level training**; you’ll need more epochs / a bigger model for perfect generalization.
* But it demonstrates the principle: **define the tokens = define the algorithmic language** → GPT-2 (with DHMM) learns the algorithm.

---

👉 Do you want me to next **extend this to sorting** (so DHMM-GPT2 learns the *language of sorting steps*, not just arithmetic)?

Alright 🚀 — let’s build the **full `DHMMTransformerEncoderDecoder` model** in PyTorch. This combines:

* **Encoder** → stacked DHMM-Transformer blocks.
* **Decoder** → stacked DHMM-Transformer blocks with **causal masking** + **cross-attention** to the encoder.
* **Cross-Attention** → decoder queries attend to encoder’s outputs, with latent DHMM states influencing attention.
* **Noise + Dithering** → applied inside both encoder and decoder blocks.
* **Heads** → produce next-token logits and latent-state logits.

---

# 📝 Full PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================
# Differentiable HMM
# =========================================================
class DifferentiableHMM(nn.Module):
    def __init__(self, num_states, embed_size):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(num_states, num_states))
        self.emitter = nn.Linear(num_states, embed_size)

    def forward(self, state_dist):
        A = self.transitions.softmax(dim=-1)
        next_state = state_dist @ A
        emission = self.emitter(next_state)
        return next_state, emission


# =========================================================
# DHMM Attention (Self + Cross)
# =========================================================
class DHMMAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_states,
                 dropout=0.1, noise=0.01, dither=0.01):
        super().__init__()
        assert embed_size % num_heads == 0
        self.h = num_heads
        self.d = embed_size // num_heads
        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)
        self.o = nn.Linear(embed_size, embed_size)
        self.hmm = DifferentiableHMM(num_states, embed_size)
        self.noise = noise
        self.dither = dither
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in, k_in, v_in, state_dist, mask=None):
        B, Lq, E = q_in.shape
        Lk = k_in.shape[1]
        Q = self.q(q_in).view(B, Lq, self.h, self.d).transpose(1, 2)
        K = self.k(k_in).view(B, Lk, self.h, self.d).transpose(1, 2)
        V = self.v(v_in).view(B, Lk, self.h, self.d).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)

        # Markov bias
        state_dist, emission = self.hmm(state_dist)
        hmm_bias = emission.mean(dim=-1, keepdim=True)
        scores = scores + hmm_bias.unsqueeze(-1)

        # Noise + Dither
        if self.noise > 0:
            scores = scores + torch.randn_like(scores) * self.noise
        if self.dither > 0:
            d = torch.randn_like(scores) * self.dither
            scores = scores + d - d.mean(dim=-1, keepdim=True)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(scores, dim=-1)
        out = (att @ V).transpose(1, 2).contiguous().view(B, Lq, E)
        return self.o(self.drop(out)), state_dist


# =========================================================
# Transformer Block (Encoder / Decoder)
# =========================================================
class DHMMBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden, num_states,
                 dropout=0.1, noise=0.01, dither=0.01, cross_attention=False):
        super().__init__()
        self.self_attn = DHMMAttention(embed_size, heads, num_states, dropout, noise, dither)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff1 = nn.Linear(embed_size, ff_hidden)
        self.ff2 = nn.Linear(ff_hidden, embed_size)
        self.drop = nn.Dropout(dropout)
        self.noise, self.dither = noise, dither
        self.cross_attention = cross_attention
        if cross_attention:
            self.cross_attn = DHMMAttention(embed_size, heads, num_states, dropout, noise, dither)
            self.ln_x = nn.LayerNorm(embed_size)

    def forward(self, x, state_dist, mask=None, enc_out=None, enc_mask=None):
        # Self-attention
        sa, state_dist = self.self_attn(x, x, x, state_dist, mask)
        x = self.ln1(x + self.drop(sa))

        # Cross-attention (for decoder only)
        if self.cross_attention and enc_out is not None:
            ca, state_dist = self.cross_attn(x, enc_out, enc_out, state_dist, enc_mask)
            x = self.ln_x(x + self.drop(ca))

        # Feedforward
        ff = F.relu(self.ff1(x))
        ff = self.ff2(ff)
        if self.noise > 0:
            ff = ff + torch.randn_like(ff) * self.noise
        if self.dither > 0:
            d = torch.randn_like(ff) * self.dither
            ff = ff + d - d.mean(dim=-1, keepdim=True)
        x = self.ln2(x + self.drop(ff))
        return x, state_dist


# =========================================================
# Heads
# =========================================================
class DHMMHead(nn.Module):
    def __init__(self, embed_size, vocab_size, num_states):
        super().__init__()
        self.lm = nn.Linear(embed_size, vocab_size)
        self.state = nn.Linear(embed_size, num_states)

    def forward(self, x):
        return self.lm(x), self.state(x)


# =========================================================
# Full Encoder–Decoder Model
# =========================================================
class DHMMTransformerEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, layers=4, heads=8, ff_hidden=1024,
                 num_states=16, max_len=512, dropout=0.1, noise=0.01, dither=0.01,
                 state_loss_weight=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_size)
        self.pos = nn.Embedding(max_len, embed_size)

        # Encoder
        self.encoder = nn.ModuleList([
            DHMMBlock(embed_size, heads, ff_hidden, num_states,
                      dropout, noise, dither, cross_attention=False)
            for _ in range(layers)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            DHMMBlock(embed_size, heads, ff_hidden, num_states,
                      dropout, noise, dither, cross_attention=True)
            for _ in range(layers)
        ])

        self.ln_f = nn.LayerNorm(embed_size)
        self.head = DHMMHead(embed_size, vocab_size, num_states)
        self.num_states, self.vocab_size, self.state_loss_weight = num_states, vocab_size, state_loss_weight

    @staticmethod
    def causal_mask(L, device):
        return torch.tril(torch.ones(L, L, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None,
                labels=None, state_labels=None):
        B, Ls = src_ids.shape
        Lt = tgt_ids.shape[1]
        device = src_ids.device

        # Embeddings
        src = self.tok(src_ids) + self.pos(torch.arange(Ls, device=device)).unsqueeze(0)
        tgt = self.tok(tgt_ids) + self.pos(torch.arange(Lt, device=device)).unsqueeze(0)

        # Masks
        cmask = self.causal_mask(Lt, device)
        if tgt_mask is not None:
            cmask = cmask * tgt_mask[:, None, None, :]
        if src_mask is not None:
            src_mask = src_mask[:, None, None, :]

        # Encoder
        state_dist = torch.full((B, self.num_states), 1.0/self.num_states, device=device)
        for blk in self.encoder:
            src, state_dist = blk(src, state_dist, src_mask)

        # Decoder
        for blk in self.decoder:
            tgt, state_dist = blk(tgt, state_dist, cmask, enc_out=src, enc_mask=src_mask)

        x = self.ln_f(tgt)
        lm_logits, state_logits = self.head(x)

        loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size),
                                      shift_labels.view(-1))
            if state_labels is not None:
                s_logits = state_logits[:, :-1, :].contiguous()
                s_labels = state_labels[:, 1:].contiguous()
                state_loss = F.cross_entropy(s_logits.view(-1, self.num_states),
                                             s_labels.view(-1))
                loss = lm_loss + self.state_loss_weight * state_loss
            else:
                loss = lm_loss

        return {"logits": lm_logits, "state_logits": state_logits, "loss": loss}
```

---

## ✅ What You Get

* A **full seq2seq Transformer** with **DHMM-enhanced attention**.
* **Encoder**: processes input sequence with DHMM latent dynamics.
* **Decoder**: autoregressively generates output sequence with **causal + cross attention**, also DHMM-enhanced.
* **Noise + Dithering** built into blocks.
* **Loss**: next-token prediction + optional latent state supervision.

---

👉 Do you want me to show a **toy training example** (like addition-as-translation: encode `"7+8="`, decode `"15"`) to demonstrate how this encoder–decoder works in practice?
