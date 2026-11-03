# -*- coding: utf-8 -*-
"""Entrena el LSTM de melodía del estilo melodic techno y lo exporta a models/techno/.

Usa las mismas piezas que tools/train_melody.py (codificación v2 relativa a la
tónica, misma red, la app la ejecuta en NumPy). Diferencias:
  - Datos: todas las melodías de library_techno (columnas melody, melody2,
    melody3 de metadata.csv), leídas como lee la app el techno (sin micro-notas,
    cuantizadas a 1/16, una voz).
  - Se aparta por canción (todas sus melodías) para validar.
  - Compara dos variantes y exporta la ganadora, reentrenada con todo:
      "techno":   solo las melodías techno
      "pretrain": primero ambient + lo-fi (library/ y library_lofi/ del repo),
                  luego afina con las techno
No lee el Sheet: usa los metadata.csv del repo. No toca los modelos ambient ni lo-fi.

Uso (desde la carpeta musicai_webapp):
    python tools/train_techno.py
    python tools/train_techno.py --holdout song_005 song_012 song_019 --epochs 200
"""

import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import train_melody as tm  # noqa: E402
from musicgen import originality, styles, theory  # noqa: E402
from musicgen.melody import ENCODING_V2  # noqa: E402
from musicgen.midi_io import read_notes  # noqa: E402

DEFAULT_HOLDOUT = ["song_005", "song_012", "song_019"]  # mayor, menor armónica, menor natural


def rows_of(path):
    with open(path, encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)
                if (r.get("Name") or "").strip()]


def song_id(name):
    digits = "".join(ch for ch in name if ch.isdigit())
    return f"song_{int(digits):03d}" if digits else None


def load_techno(library, columns, read_options):
    """-> [Song] one per melody file; Song.song = song_XXX (for the holdout)."""
    songs = []
    for row in rows_of(os.path.join(library, "metadata.csv")):
        sid = song_id(row["Name"])
        try:
            scale = theory.scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
        except Exception as e:  # noqa: BLE001
            print(f"  {sid}: escala no reconocida ({e}), se omite")
            continue
        for col in columns:
            fname = row.get(col, "")
            path = os.path.join(library, sid, fname)
            if not fname or not os.path.isfile(path):
                continue
            notes = read_notes(path, top_voice_only=True, monophonic=True, **read_options)
            if len(notes) < 4:
                continue
            s = tm.Song(sid if col == "melody" else f"{sid}/{col}", notes, scale, "techno")
            s.song, s.mood = sid, row.get("mood", "")
            songs.append(s)
    return songs


def load_other(library, prefix, source):
    """Ambient / lo-fi melodies bundled in the repo (for the 'pretrain' variant)."""
    songs = []
    for row in rows_of(os.path.join(library, "metadata.csv")):
        if row.get("source", "").lower() == "generated":
            continue
        sid = song_id(row["Name"])
        folder = os.path.join(library, sid) if os.path.isdir(os.path.join(library, sid)) \
            else os.path.join(library, sid.replace("_", ""))
        path = os.path.join(folder, row.get("melody", ""))
        if not row.get("melody") or not os.path.isfile(path):
            continue
        try:
            scale = theory.scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
        except Exception:  # noqa: BLE001
            continue
        songs.append(tm.Song(f"{prefix}{sid}", read_notes(path), scale, source))
    return songs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", nargs="+", default=DEFAULT_HOLDOUT, help="canciones apartadas para validar")
    ap.add_argument("--octave-center", type=float, default=tm.DEFAULT_CENTER)
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(REPO, "models", "techno"))
    args = ap.parse_args()
    center = None if args.octave_center < 0 else args.octave_center
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    cfg = styles.get("worakls")
    techno = load_techno(cfg["library"], cfg["melody_columns"], cfg["pattern_options"]["read_options"])
    songs = sorted({s.song for s in techno})
    missing = [h for h in args.holdout if h not in songs]
    if missing:
        sys.exit(f"Canciones de --holdout que no existen: {missing}")
    pre = [] if args.no_pretrain else (
        load_other(os.path.join(REPO, "library"), "ambient_", "ambient")
        + load_other(os.path.join(REPO, "library_lofi"), "lofi_", "lofi"))
    print(f"Techno: {len(techno)} melodías de {len(songs)} canciones · pretrain: {len(pre)} melodías")

    enc, shifts = {}, {}
    for s in techno + pre:
        enc[s.name], shifts[s.name] = tm.encode(s, center)
    train_names = [s.name for s in techno if s.song not in args.holdout]
    val_names = [s.name for s in techno if s.song in args.holdout]
    print(f"Entrenamiento {len(train_names)} melodías · validación {len(val_names)} ({' '.join(args.holdout)})")

    library = {s.name: originality.make_entry(s.name, s.notes, s.scale) for s in techno}

    experiments = [("techno", [])] + ([("pretrain", pre)] if pre else [])
    results = {}
    for label, pre_songs in experiments:
        print(f"\n=== Variante '{label}'")
        torch.manual_seed(args.seed)
        s2i = tm.build_vocab([enc[n] for n in train_names + val_names] + [enc[s.name] for s in pre_songs])
        model = tm.MelodyLSTM(len(s2i), hidden_size=args.hidden, dropout=args.dropout)
        info = {}
        if pre_songs:
            tm.fit(model, tm.windows([enc[s.name] for s in pre_songs], s2i), epochs=args.pretrain_epochs,
                   label="ambient+lofi")
            info["pretrain_epochs"] = args.pretrain_epochs
        info.update(tm.fit(model, tm.windows([enc[n] for n in train_names], s2i),
                           tm.windows([enc[n] for n in val_names], s2i), epochs=args.epochs,
                           lr=5e-4 if pre_songs else 1e-3, patience=args.patience, label=label))
        info["copies"] = dict(tm.copy_rate(model, s2i, techno, library, n=args.samples, seed=args.seed))
        info["copy_rate"] = info["copies"].get("COPIA", 0) / max(1, sum(info["copies"].values()))
        print(f"  [{label}] {args.samples} melodías generadas: {info['copies']}")
        results[label] = info

    ranked = sorted(results, key=lambda k: results[k]["val_loss"])
    winner = ranked[0]
    if len(ranked) > 1 and results[winner]["copy_rate"] > results[ranked[1]]["copy_rate"] + 0.10:
        winner = ranked[1]
    print("\n" + "=" * 70 + "\nCOMPARACIÓN\n" + "=" * 70)
    print(f"{'variante':10s} {'mejor época':>11s} {'val_loss':>9s} {'val_acc':>8s} {'% copias':>9s}")
    for k, r in results.items():
        mark = "  <- ganadora" if k == winner else ""
        print(f"{k:10s} {r['best_epoch']:11d} {r['val_loss']:9.3f} {r['val_acc']:8.2f} {r['copy_rate']:9.0%}{mark}")

    print(f"\n=== Modelo final: variante '{winner}' con las {len(techno)} melodías")
    torch.manual_seed(args.seed)
    pre_songs = pre if winner == "pretrain" else []
    all_names = [s.name for s in techno]
    s2i = tm.build_vocab([enc[n] for n in all_names] + [enc[s.name] for s in pre_songs])
    model = tm.MelodyLSTM(len(s2i), hidden_size=args.hidden, dropout=args.dropout)
    if pre_songs:
        tm.fit(model, tm.windows([enc[s.name] for s in pre_songs], s2i), epochs=args.pretrain_epochs,
               label="ambient+lofi")
    final_epochs = max(1, results[winner]["best_epoch"])
    tm.fit(model, tm.windows([enc[n] for n in all_names], s2i), epochs=final_epochs,
           lr=5e-4 if pre_songs else 1e-3, label="final")
    final_copies = tm.copy_rate(model, s2i, techno, library, n=args.samples, seed=args.seed + 1)
    print(f"  [final] {args.samples} melodías generadas: {dict(final_copies)}")

    report = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "style": "worakls",
        "winner": winner,
        "settings": {"encoding": ENCODING_V2, "sequence_length": tm.SEQ_LEN, "octave_center": center,
                     "holdout": args.holdout, "hidden": args.hidden, "dropout": args.dropout,
                     "final_epochs": final_epochs, "pretrain_epochs": args.pretrain_epochs if pre_songs else 0,
                     "techno_melodies": all_names, "pretrain_melodies": [s.name for s in pre_songs]},
        "comparison": {k: {kk: vv for kk, vv in r.items() if kk != "history"} for k, r in results.items()},
        "final_copies": dict(final_copies),
        "octave_shifts": {k: v for k, v in shifts.items() if k.startswith("song_")},
        "vocab_size": len(s2i),
    }
    tm.export(model, s2i, args.out, report)
    print(f"\nExportado a {os.path.abspath(args.out)} (vocabulario {len(s2i)} símbolos) en {time.time() - t0:.0f} s.")


if __name__ == "__main__":
    main()
