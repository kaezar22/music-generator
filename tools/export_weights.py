"""Convierte melody_lstm.pt (PyTorch) a models/melody_lstm_weights.npz.

Solo hace falta si vuelves a entrenar el modelo en el notebook. Requiere torch
(localmente, no en Streamlit Cloud). Copia también melody_vocab.json a models/.

Uso:
    python tools/export_weights.py ..\\AI_MUSIC_Project\\melody_lstm.pt
"""
import os
import sys

import numpy as np
import torch

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "AI_MUSIC_Project", "melody_lstm.pt")
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(here, "models", "melody_lstm_weights.npz")
sd = torch.load(src, map_location="cpu")
np.savez_compressed(out, **{k.replace(".", "__"): v.numpy() for k, v in sd.items()})
print(f"Escrito {out}")
