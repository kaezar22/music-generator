# -*- coding: utf-8 -*-
"""Entrena el LSTM de melodía con el dataset lo-fi y lo exporta para la app.

Reemplaza las celdas de entrenamiento del notebook. Pasos:
  1. Valida el dataset (mismo chequeo que tools/validate_lofi_dataset.py) y
     se detiene si hay errores.
  2. Codifica cada melodía en semicorcheas con grados de su propia escala,
     relativos a la tónica (codificación v2), y normaliza la octava.
  3. Aparta 3 canciones para validar y entrena dos variantes:
       - "lofi":     solo las canciones lo-fi
       - "pretrain": primero las melodías ambient, luego afina con las lo-fi
     y compara pérdida en las canciones apartadas y % de copias.
  4. Reentrena la variante ganadora con las 20 canciones y la exporta a
     models/lofi/ (pesos .npz + vocabulario + informe), sin tocar el modelo
     ambient de la app.

Requiere: torch, numpy, mido (+ gspread y google-auth para leer el Sheet).

Uso (desde la carpeta musicai_webapp):
    python tools/train_melody.py
    python tools/train_melody.py --no-pretrain            # solo la variante lo-fi
    python tools/train_melody.py --holdout song_003 song_009 song_016
    python tools/train_melody.py --csv lofi.csv --ambient-csv ..\\AI_MUSIC_Project\\music_metadata_full.csv
"""

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import validate_lofi_dataset as vld  # noqa: E402
from musicgen import originality, theory  # noqa: E402
from musicgen.melody import ENCODING_V2, pitch_to_symbol, symbols_to_events  # noqa: E402
from musicgen.midi_io import read_notes  # noqa: E402
from musicgen.moods import MOOD_GROUPS, mood_group  # noqa: E402

STEP = 0.25
SEQ_LEN = 32
AMBIENT_GID = 1183065346
DEFAULT_HOLDOUT = ["song_003", "song_009", "song_016"]  # dorian, menor armónica, mixolidia
DEFAULT_CENTER = 38  # grado 3 de la 5ª octava sobre la tónica (≈ registro medio-agudo)


# ------------------------------------------------------------------ datos
class Song:
    def __init__(self, name, notes, scale, source):
        self.name, self.notes, self.scale, self.source = name, notes, scale, source


def load_lofi(args):
    by_song, folders, rep, _ = vld.run_validation(args)
    errors = rep.count("ERROR")
    if errors and not args.skip_validation:
        print(f"La validación encontró {errors} errores. Corre tools/validate_lofi_dataset.py para verlos.")
        sys.exit(1)
    print(f"Validación: {errors} errores, {rep.count('AVISO')} avisos.")
    songs = []
    for name in sorted(set(by_song) & set(folders)):
        p = by_song[name]
        path = os.path.join(folders[name], f"lofi_melody_{p['num']:03d}.mid")
        songs.append(Song(name, read_notes(path), p["scale"], "lofi"))
        songs[-1].mood = p["row"]["mood"]
    return songs


def load_ambient(args):
    """Melodías ambient de AI_MUSIC_Project (metadata en la pestaña ambient).
    Si el Sheet tiene una columna 'source', se omiten las filas 'generated'."""
    ns = argparse.Namespace(csv=args.ambient_csv, credentials=args.credentials, gid=AMBIENT_GID)
    try:
        rows = vld.read_sheet_rows(ns, vld.Report())
    except Exception as e:  # noqa: BLE001
        print(f"No se pudo leer la metadata ambient ({e}); se omite la variante 'pretrain'.")
        return []
    songs, skipped = [], 0
    for _, raw in rows:
        row = {k: str(v).strip() for k, v in raw.items()}
        if row.get("source", "").lower() == "generated":
            skipped += 1
            continue
        num = vld.song_number(row.get("Name", ""))
        if num is None or not row.get("melody"):
            continue
        path = os.path.join(args.ambient_dir, f"song_{num:03d}", row["melody"])
        if not os.path.isfile(path):
            continue
        try:
            scale = theory.scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
            songs.append(Song(f"ambient_{num:03d}", read_notes(path), scale, "ambient"))
        except Exception:  # noqa: BLE001
            continue
    msg = f"Ambient: {len(songs)} melodías"
    print(msg + (f" ({skipped} generadas por la app omitidas)" if skipped else ""))
    return songs


def encode(song, center):
    """-> lista de símbolos por semicorchea (v2), con la octava normalizada."""
    syms = [pitch_to_symbol(p, song.scale, ENCODING_V2) for _, p, _ in song.notes]
    idx = [int(s.rstrip("+-")) for s in syms]
    shift = 0
    if center is not None and idx:
        shift = round((center - (min(idx) + max(idx)) / 2) / 7)
    total = round(max(o + d for o, _, d in song.notes) / STEP)
    grid = ["r"] * total
    for (o, _, d), s, i in zip(song.notes, syms, idx):
        start, n = round(o / STEP), max(1, round(d / STEP))
        sym = f"{i + 7 * shift}{s[len(str(i)):]}"
        for k in range(n):
            if start + k < total:
                grid[start + k] = sym if k == 0 else "_"
    return grid, shift


def build_vocab(encoded):
    symbols = {s for seq in encoded for s in seq}
    notes = sorted((s for s in symbols if s not in ("_", "r")), key=lambda s: (int(s.rstrip("+-")), s))
    vocab = ["_", "r"] + notes
    return {s: i for i, s in enumerate(vocab)}


def windows(encoded, s2i):
    """Cada canción va precedida de SEQ_LEN silencios, igual que la semilla que
    usa la app al generar, y las ventanas nunca cruzan de una canción a otra."""
    xs, ys = [], []
    for seq in encoded:
        ids = [s2i["r"]] * SEQ_LEN + [s2i[s] for s in seq]
        for i in range(len(seq)):
            xs.append(ids[i:i + SEQ_LEN])
            ys.append(ids[i + SEQ_LEN])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# ------------------------------------------------------------------ modelo
class MelodyLSTM(nn.Module):
    """Misma arquitectura que el notebook: la app la ejecuta en NumPy."""

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(self.dropout(out[:, -1, :]))


def run_epoch(model, x, y, opt=None, batch=64):
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, correct = 0.0, 0
    order = torch.randperm(len(x)) if train else torch.arange(len(x))
    with torch.set_grad_enabled(train):
        for i in range(0, len(x), batch):
            b = order[i:i + batch]
            logits = model(x[b])
            loss = crit(logits, y[b])
            if train:
                opt.zero_grad()
                (loss / len(b)).backward()
                opt.step()
            total += loss.item()
            correct += (logits.argmax(1) == y[b]).sum().item()
    return total / len(x), correct / len(x)


def fit(model, train, val=None, epochs=150, lr=1e-3, patience=15, label=""):
    """Entrena; con val hace early stopping y devuelve el mejor estado."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = (float("inf"), 0, None, 0.0)
    history = []
    for ep in range(1, epochs + 1):
        tl, ta = run_epoch(model, *train, opt)
        rec = {"epoch": ep, "train_loss": tl, "train_acc": ta}
        if val is not None:
            vl, va = run_epoch(model, *val)
            rec.update(val_loss=vl, val_acc=va)
            if vl < best[0]:
                best = (vl, ep, copy.deepcopy(model.state_dict()), va)
            elif ep - best[1] >= patience:
                history.append(rec)
                break
        history.append(rec)
        if ep == 1 or ep % 10 == 0:
            extra = f"  val_loss={rec['val_loss']:.3f} val_acc={rec['val_acc']:.2f}" if val is not None else ""
            print(f"  [{label}] época {ep:3d}  train_loss={tl:.3f} train_acc={ta:.2f}{extra}")
    if val is not None:
        model.load_state_dict(best[2])
        print(f"  [{label}] mejor época {best[1]}: val_loss={best[0]:.3f} val_acc={best[3]:.2f}")
        return {"best_epoch": best[1], "val_loss": best[0], "val_acc": best[3], "history": history}
    return {"history": history}


# ------------------------------------------------------------------ evaluación
def generate(model, s2i, steps, temperature, rng):
    i2s = {i: s for s, i in s2i.items()}
    ctx = [s2i["r"]] * SEQ_LEN
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            logits = model(torch.tensor([ctx[-SEQ_LEN:]]))[0] / temperature
            p = torch.softmax(logits, 0).numpy().astype(np.float64)
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            out.append(i2s[nxt])
            ctx.append(nxt)
    return out


def copy_rate(model, s2i, lofi, library, n=20, temperature=0.9, seed=0):
    rng = np.random.default_rng(seed)
    verdicts = Counter()
    for k in range(n):
        song = lofi[k % len(lofi)]
        events = symbols_to_events(generate(model, s2i, 16 * 16, temperature, rng), song.scale, ENCODING_V2)
        if len(events) < 8:
            verdicts["VACÍA"] += 1
            continue
        verdicts[originality.check_melody(events, song.scale, library)["verdict"]] += 1
    return verdicts


def export(model, s2i, out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    np.savez_compressed(os.path.join(out_dir, "melody_lstm_weights.npz"),
                        **{k.replace(".", "__"): v.cpu().numpy() for k, v in sd.items()})
    with open(os.path.join(out_dir, "melody_vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"symbol_to_id": s2i, "sequence_length": SEQ_LEN, "encoding": ENCODING_V2,
                   "octave_center": report["settings"]["octave_center"]}, f, indent=2)
    with open(os.path.join(out_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vld.add_source_args(ap)
    ap.add_argument("--ambient-dir", default=os.path.join(REPO, "..", "AI_MUSIC_Project"))
    ap.add_argument("--ambient-csv", help="CSV de la pestaña ambient (por defecto se lee el Sheet)")
    ap.add_argument("--no-pretrain", action="store_true", help="entrenar solo la variante lo-fi")
    ap.add_argument("--holdout", nargs="+", default=DEFAULT_HOLDOUT, help="canciones apartadas para validar")
    ap.add_argument("--octave-center", type=float, default=DEFAULT_CENTER,
                    help="centro común al que se lleva cada melodía (en grados); usa -1 para no normalizar")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--samples", type=int, default=20, help="melodías generadas para medir copias")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "models", "lofi"))
    args = ap.parse_args()
    center = None if args.octave_center < 0 else args.octave_center

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    t0 = time.time()

    lofi = load_lofi(args)
    ambient = [] if args.no_pretrain else load_ambient(args)
    missing = [h for h in args.holdout if h not in {s.name for s in lofi}]
    if missing:
        sys.exit(f"Canciones de --holdout que no existen: {missing}")

    enc = {}
    shifts = {}
    for s in lofi + ambient:
        enc[s.name], shifts[s.name] = encode(s, center)
    train_names = [s.name for s in lofi if s.name not in args.holdout]
    print(f"\nLo-fi: {len(lofi)} canciones · entrenamiento {len(train_names)} · validación {args.holdout}")
    if center is not None:
        moved = {k: v for k, v in shifts.items() if v and k.startswith("song_")}
        print(f"Octava normalizada (centro {center:g}): {len(moved)} melodías desplazadas "
              + ", ".join(f"{k} {v:+d}" for k, v in sorted(moved.items())))

    library = {}
    for s in lofi + ambient:
        library[s.name] = originality.make_entry(s.name, s.notes, s.scale)

    experiments = [("lofi", [])] + ([("pretrain", ambient)] if ambient else [])
    results = {}
    for label, pre in experiments:
        print(f"\n=== Variante '{label}'")
        s2i = build_vocab([enc[n] for n in train_names + args.holdout] + [enc[s.name] for s in pre])
        model = MelodyLSTM(len(s2i), hidden_size=args.hidden, dropout=args.dropout)
        info = {}
        if pre:
            fit(model, windows([enc[s.name] for s in pre], s2i), epochs=args.pretrain_epochs, label="ambient")
            info["pretrain_epochs"] = args.pretrain_epochs
        train = windows([enc[n] for n in train_names], s2i)
        val = windows([enc[n] for n in args.holdout], s2i)
        info.update(fit(model, train, val, epochs=args.epochs, lr=5e-4 if pre else 1e-3,
                        patience=args.patience, label=label))
        info["copies"] = dict(copy_rate(model, s2i, lofi, library, n=args.samples, seed=args.seed))
        n = sum(info["copies"].values())
        info["copy_rate"] = info["copies"].get("COPIA", 0) / n
        print(f"  [{label}] {args.samples} melodías generadas: {info['copies']}")
        results[label] = info

    # ganador: menor pérdida en canciones apartadas, salvo que copie bastante más
    ranked = sorted(results, key=lambda k: results[k]["val_loss"])
    winner = ranked[0]
    if len(ranked) > 1 and results[winner]["copy_rate"] > results[ranked[1]]["copy_rate"] + 0.10:
        winner = ranked[1]

    print("\n" + "=" * 70 + "\nCOMPARACIÓN\n" + "=" * 70)
    print(f"{'variante':10s} {'mejor época':>11s} {'val_loss':>9s} {'val_acc':>8s} {'% copias':>9s}")
    for k, r in results.items():
        mark = "  <- ganadora" if k == winner else ""
        print(f"{k:10s} {r['best_epoch']:11d} {r['val_loss']:9.3f} {r['val_acc']:8.2f} {r['copy_rate']:9.0%}{mark}")

    # reentrenar la ganadora con las 20 canciones, el mismo número de épocas
    print(f"\n=== Modelo final: variante '{winner}' con las {len(lofi)} canciones")
    torch.manual_seed(args.seed)
    pre = ambient if winner == "pretrain" else []
    all_names = [s.name for s in lofi]
    s2i = build_vocab([enc[n] for n in all_names] + [enc[s.name] for s in pre])
    model = MelodyLSTM(len(s2i), hidden_size=args.hidden, dropout=args.dropout)
    if pre:
        fit(model, windows([enc[s.name] for s in pre], s2i), epochs=args.pretrain_epochs, label="ambient")
    final_epochs = results[winner]["best_epoch"]
    fit(model, windows([enc[n] for n in all_names], s2i), epochs=final_epochs,
        lr=5e-4 if pre else 1e-3, label="final")
    final_copies = copy_rate(model, s2i, lofi, library, n=args.samples, seed=args.seed + 1)
    print(f"  [final] {args.samples} melodías generadas: {dict(final_copies)}")

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "winner": winner,
        "settings": {"encoding": ENCODING_V2, "sequence_length": SEQ_LEN, "octave_center": center,
                     "holdout": args.holdout, "hidden": args.hidden, "dropout": args.dropout,
                     "final_epochs": final_epochs, "pretrain_epochs": args.pretrain_epochs if pre else 0,
                     "lofi_songs": all_names, "ambient_songs": [s.name for s in pre],
                     "mood_conditioning": False},
        "comparison": {k: {kk: vv for kk, vv in r.items() if kk != "history"} for k, r in results.items()},
        "final_copies": dict(final_copies),
        "octave_shifts": {k: v for k, v in shifts.items() if k.startswith("song_")},
        "mood_groups": MOOD_GROUPS,
        "song_mood_groups": {s.name: mood_group(s.mood) for s in lofi},
        "vocab_size": len(s2i),
    }
    export(model, s2i, args.out, report)
    print(f"\nExportado a {os.path.abspath(args.out)} (vocabulario {len(s2i)} símbolos) "
          f"en {time.time() - t0:.0f} s.")
    print("La app ambient no cambia: su modelo sigue en models/melody_lstm_weights.npz.")


if __name__ == "__main__":
    main()
