"""Melody generation with the project's LSTM, re-implemented in NumPy.

The weights come from melody_lstm.pt (trained in the notebook) and were
exported to models/melody_lstm_weights.npz, so the web app does not need
PyTorch. The forward pass matches torch.nn.LSTM (gate order i, f, g, o) and
the generation loop matches the notebook: every step the model reads the last
SEQUENCE_LENGTH symbols, starting from a context of rests.
"""

import json
import os

import numpy as np

from .theory import STEP

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MelodyModel:
    def __init__(self, models_dir=MODELS_DIR):
        with open(os.path.join(models_dir, "melody_vocab.json")) as f:
            vocab = json.load(f)
        self.symbol_to_id = vocab["symbol_to_id"]
        self.id_to_symbol = {int(v): k for k, v in self.symbol_to_id.items()}
        self.seq_len = vocab["sequence_length"]
        w = np.load(os.path.join(models_dir, "melody_lstm_weights.npz"))
        self.emb = w["embedding__weight"].astype(np.float64)
        self.layers = []
        l = 0
        while f"lstm__weight_ih_l{l}" in w:
            self.layers.append((
                w[f"lstm__weight_ih_l{l}"].astype(np.float64),
                w[f"lstm__weight_hh_l{l}"].astype(np.float64),
                (w[f"lstm__bias_ih_l{l}"] + w[f"lstm__bias_hh_l{l}"]).astype(np.float64),
            ))
            l += 1
        self.fc_w = w["fc__weight"].astype(np.float64)
        self.fc_b = w["fc__bias"].astype(np.float64)
        self.vocab_size = len(self.symbol_to_id)

    def logits(self, ids):
        """ids: sequence of token ids -> logits for the next token."""
        x = self.emb[np.asarray(ids)]  # (T, E)
        for w_ih, w_hh, b in self.layers:
            hidden = w_hh.shape[1]
            h = np.zeros(hidden)
            c = np.zeros(hidden)
            pre = x @ w_ih.T + b  # (T, 4H)
            outs = np.empty((len(ids), hidden))
            for t in range(len(ids)):
                g = pre[t] + w_hh @ h
                i_g = _sigmoid(g[:hidden])
                f_g = _sigmoid(g[hidden:2 * hidden])
                g_g = np.tanh(g[2 * hidden:3 * hidden])
                o_g = _sigmoid(g[3 * hidden:])
                c = f_g * c + i_g * g_g
                h = o_g * np.tanh(c)
                outs[t] = h
            x = outs
        return self.fc_w @ x[-1] + self.fc_b

    def generate_symbols(self, num_steps, temperature=0.9, rng=None, seed_symbols=None):
        rng = rng or np.random.default_rng()
        if seed_symbols is None:
            seed_symbols = ["r"] * self.seq_len
        context = [self.symbol_to_id[s] for s in seed_symbols]
        out = []
        for _ in range(num_steps):
            z = self.logits(context[-self.seq_len:]) / max(temperature, 1e-3)
            z = z - z.max()
            p = np.exp(z)
            p /= p.sum()
            nxt = int(rng.choice(self.vocab_size, p=p))
            out.append(self.id_to_symbol[nxt])
            context.append(nxt)
        return out


def symbol_to_pitch(symbol, scale):
    alteration = 0
    core = symbol
    if symbol.endswith("+"):
        alteration, core = 1, symbol[:-1]
    elif symbol.endswith("-"):
        alteration, core = -1, symbol[:-1]
    idx = int(core)
    return scale[idx % 7] + 12 * (idx // 7) + alteration


def symbols_to_events(symbols, scale):
    """Grid symbols -> [(start_beat, pitch, dur_beats)] decoded in the target scale."""
    events = []
    cur = None
    for k, s in enumerate(symbols):
        if s in ("r", "/"):
            if cur:
                events.append(tuple(cur))
            cur = None
        elif s == "_":
            if cur:
                cur[2] += STEP
        else:
            if cur:
                events.append(tuple(cur))
            cur = [k * STEP, symbol_to_pitch(s, scale), STEP]
    if cur:
        events.append(tuple(cur))
    return events
