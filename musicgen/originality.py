"""Melody originality check (same method as check_originality.py).

Each melody becomes a sequence of steps (interval to next note, 16ths to next
onset), so copies are found even when transposed. With the scales known it is
also compared in scale degrees, which catches a melody re-decoded in another
mode (the LSTM does this). Only the melody is compared.
"""

import glob
import io
import os
import re
import zipfile

from .midi_io import read_notes
from .theory import STEP, scale_notes_ordered

NGRAM = 6
COPY_RUN = 24         # >= 24 identical consecutive notes (~4 bars) -> COPIA
COPY_COVERAGE = 0.60  # >= 60 % of the melody taken from ONE song -> COPIA
WARN_RUN = 12         # >= 12 notes (~2 bars) -> SIMILAR
WARN_COVERAGE = 0.30
RANK = {"ORIGINAL": 0, "SIMILAR": 1, "COPIA": 2}

SONG_RE = re.compile(r"song_(\d+)", re.I)


def diatonic_index(pitch, scale):
    pc, octv = pitch % 12, pitch // 12
    if pc in scale:
        return scale.index(pc) + 7 * octv
    for delta in (-1, 1, -2, 2):
        if (pc + delta) % 12 in scale:
            return scale.index((pc + delta) % 12) + 7 * octv
    return 7 * octv


def tokens(notes, scale=None):
    if len(notes) < 2:
        return []
    vals = [diatonic_index(p, scale) for _, p, _ in notes] if scale else [p for _, p, _ in notes]
    return [(vals[i + 1] - vals[i], min(round((notes[i + 1][0] - notes[i][0]) / STEP), 32))
            for i in range(len(notes) - 1)]


def longest_common_run(a, b):
    best = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def coverage(cand, ref, n=NGRAM):
    if len(cand) < n:
        return 0.0
    grams = {tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)}
    covered = [False] * len(cand)
    for i in range(len(cand) - n + 1):
        if tuple(cand[i:i + n]) in grams:
            for k in range(i, i + n):
                covered[k] = True
    return sum(covered) / len(cand)


# ------------------------------------------------------------------ library
def make_entry(name, notes, scale):
    return {
        "name": name,
        "scale": scale,
        "notes": notes,
        "chrom": tokens(notes),
        "diat": tokens(notes, scale) if scale else None,
        "abs": [(round(o / STEP), p) for o, p, _ in notes],
    }


def _scale_for(num, metadata_by_num):
    row = metadata_by_num.get(num)
    if not row:
        return None
    try:
        return scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
    except Exception:
        return None


def load_folder_library(root, metadata_by_num):
    """{song_name: entry} from root/song_XXX/melody*.mid."""
    lib = {}
    for path in sorted(glob.glob(os.path.join(root, "song_*", "melody*.mid"))):
        folder = os.path.basename(os.path.dirname(path))
        m = SONG_RE.search(folder)
        if not m:
            continue
        num = int(m.group(1))
        lib[f"song_{num:03d}"] = make_entry(f"song_{num:03d}", read_notes(path), _scale_for(num, metadata_by_num))
    return lib


def load_zip_library(zip_bytes, metadata_by_num):
    """Accepts a zip containing song_XXX folders (at any depth) with melody*.mid,
    or loose melody_XXX.mid files."""
    lib = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            base = name.rsplit("/", 1)[-1].lower()
            if info.is_dir() or not base.startswith("melody") or not base.endswith((".mid", ".midi")):
                continue
            m = SONG_RE.search(name) or re.search(r"melody_(\d+)", base)
            if not m:
                continue
            num = int(m.group(1))
            try:
                notes = read_notes(z.read(info))
            except Exception:
                continue
            lib[f"song_{num:03d}"] = make_entry(f"song_{num:03d}", notes, _scale_for(num, metadata_by_num))
    return lib


# ------------------------------------------------------------------ check
def _verdict(r, n_tokens):
    if r["identical_abs"]:
        return "COPIA", "copia EXACTA (mismas notas y ritmo)"
    if r["identical_shape"]:
        return "COPIA", f"copia transpuesta / con cambio de modo ({r['mode']})"
    run, cov = r["run"], r["coverage"]
    detail = f"{cov:.0%} coincidente, frase copiada más larga de {run} notas"
    if (cov >= COPY_COVERAGE
            or run >= 0.5 * r["ref_len"]                      # contiene media canción existente
            or run >= max(WARN_RUN + 1, 0.5 * n_tokens)       # la mitad de la nueva es una sola cita
            or (run >= COPY_RUN and cov >= WARN_COVERAGE)):
        return "COPIA", detail
    if cov >= WARN_COVERAGE or run >= WARN_RUN:
        return "SIMILAR", detail
    return "ORIGINAL", detail


def check_melody(notes, scale, library):
    """library: {name: entry}. Returns dict with verdict, reason, closest, rows."""
    cand = make_entry("candidate", notes, scale)
    n = len(cand["chrom"])
    rows = []
    for name, ref in library.items():
        if not ref["chrom"] or not n:
            continue
        run = longest_common_run(cand["chrom"], ref["chrom"])
        cov = coverage(cand["chrom"], ref["chrom"])
        mode = "cromático"
        if cand["diat"] and ref["diat"]:
            drun = longest_common_run(cand["diat"], ref["diat"])
            dcov = coverage(cand["diat"], ref["diat"])
            if (drun, dcov) > (run, cov):
                run, cov, mode = drun, dcov, "diatónico"
        r = {
            "song": name, "run": run, "coverage": cov, "mode": mode, "ref_len": len(ref["chrom"]),
            "identical_abs": cand["abs"] == ref["abs"],
            "identical_shape": cand["chrom"] == ref["chrom"]
            or (cand["diat"] is not None and cand["diat"] == ref["diat"]),
        }
        r["verdict"], r["why"] = _verdict(r, n)
        rows.append(r)
    rows.sort(key=lambda r: (RANK[r["verdict"]], r["coverage"], r["run"]), reverse=True)
    if not rows:
        return {"verdict": "ORIGINAL", "reason": "biblioteca vacía", "closest": None, "rows": [], "n_notes": len(notes)}
    top = rows[0]
    return {"verdict": top["verdict"], "reason": top["why"], "closest": top["song"],
            "rows": rows, "n_notes": len(notes)}
