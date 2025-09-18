Perfect 🚀 — let’s tie everything together into **one runnable script**:

* **CartPole Tokenizer & Dataset** → converts Gym episodes into token sequences.
* **Pure DHMM-GPT2** → the model we just built.
* **Training loop** → teaches the model to predict the next token.
* **Inference loop** → lets the trained model control CartPole.

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
