import torch, math
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Differentiable Hidden Markov Model Core
# =========================================================
class DifferentiableHMM(nn.Module):
    def __init__(self, num_states, embed_size):
        super().__init__()
        self.transitions = nn.Parameter(torch.randn(num_states, num_states))
        self.emitter = nn.Linear(num_states, embed_size)

    def forward(self, state_dist):
        trans_probs = self.transitions.softmax(dim=-1)         # S,S
        next_state = state_dist @ trans_probs                  # B,S
        emission = self.emitter(next_state)                    # B,E
        return next_state, emission


# =========================================================
# DHMM Attention (Markov-biased softmax + noise + dithering)
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

        self.noise = noise
        self.dither = dither
        self.drop = nn.Dropout(dropout)
        self.hmm = DifferentiableHMM(num_states, embed_size)

    def forward(self, x, state_dist, mask):
        B, L, E = x.shape
        Q = self.q(x).view(B, L, self.h, self.d).transpose(1, 2)
        K = self.k(x).view(B, L, self.h, self.d).transpose(1, 2)
        V = self.v(x).view(B, L, self.h, self.d).transpose(1, 2)

        energy = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d)     # B,h,L,L

        # evolve HMM and inject into attention scores
        state_dist, emission = self.hmm(state_dist)                # B,S , B,E
        hmm_bias = emission.mean(dim=-1, keepdim=True)             # B,1
        energy = energy + hmm_bias.unsqueeze(-1)                   # bias all tokens

        # noise (mutation)
        if self.noise > 0:
            energy = energy + torch.randn_like(energy) * self.noise
        # dithering (filtering)
        if self.dither > 0:
            d = torch.randn_like(energy) * self.dither
            energy = energy + d - d.mean(dim=-1, keepdim=True)

        energy = energy.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(energy, dim=-1)
        out = (att @ V).transpose(1, 2).contiguous().view(B, L, E)

        return self.o(self.drop(out)), state_dist


# =========================================================
# DHMM Transformer Block
# =========================================================
class DHMMBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden, num_states,
                 dropout=0.1, noise=0.01, dither=0.01):
        super().__init__()
        self.attn = DHMMAttention(embed_size, heads, num_states, dropout, noise, dither)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ff1 = nn.Linear(embed_size, ff_hidden)
        self.ff2 = nn.Linear(ff_hidden, embed_size)
        self.drop = nn.Dropout(dropout)
        self.noise = noise
        self.dither = dither

    def forward(self, x, state_dist, mask):
        attn_out, state_dist = self.attn(x, state_dist, mask)
        x = self.ln1(x + self.drop(attn_out))

        ff = F.relu(self.ff1(x))
        ff = self.ff2(ff)

        # noise + dithering in FF
        if self.noise > 0:
            ff = ff + torch.randn_like(ff) * self.noise
        if self.dither > 0:
            d = torch.randn_like(ff) * self.dither
            ff = ff + d - d.mean(dim=-1, keepdim=True)

        x = self.ln2(x + self.drop(ff))
        return x, state_dist


# =========================================================
# DHMM Head (LM + state head)
# =========================================================
class DHMMHead(nn.Module):
    def __init__(self, embed_size, vocab_size, num_states):
        super().__init__()
        self.lm = nn.Linear(embed_size, vocab_size)
        self.state = nn.Linear(embed_size, num_states)

    def forward(self, x):
        return self.lm(x), self.state(x)


# =========================================================
# Full DHMM-GPT Model
# =========================================================
class DHMMGPT(nn.Module):
    def __init__(self, vocab_size, embed_size=256, layers=6, heads=8,
                 ff_hidden=1024, num_states=16, max_len=512,
                 dropout=0.1, noise=0.01, dither=0.01, state_loss_weight=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_size)
        self.pos = nn.Embedding(max_len, embed_size)
        self.blocks = nn.ModuleList([
            DHMMBlock(embed_size, heads, ff_hidden, num_states, dropout, noise, dither)
            for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(embed_size)
        self.head = DHMMHead(embed_size, vocab_size, num_states)
        self.num_states = num_states
        self.vocab_size = vocab_size
        self.state_loss_weight = state_loss_weight

    @staticmethod
    def causal_mask(L, device):
        return torch.tril(torch.ones(L, L, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids, labels=None, state_labels=None, attention_mask=None):
        B, L = input_ids.shape
        device = input_ids.device
        x = self.tok(input_ids) + self.pos(torch.arange(L, device=device)).unsqueeze(0)

        mask = self.causal_mask(L, device)
        if attention_mask is not None:
            am = attention_mask[:, None, None, :].to(mask.dtype)
            mask = mask * am
        mask = mask.expand(B, -1, -1, -1)

        state_dist = torch.full((B, self.num_states), 1.0 / self.num_states, device=device)

        for blk in self.blocks:
            x, state_dist = blk(x, state_dist, mask)

        x = self.ln_f(x)
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
