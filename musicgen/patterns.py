"""Bass / arpeggio patterns learned from the songs in library/.

Every library song has a 4-bar bass and arpeggio loop with one chord per bar.
Each note is stored as its position inside the bar plus its *chord degree*
(0 = root, 2 = third, 4 = fifth, 1/3/5/6 = passing tones, counted in scale
steps above the chord root). To use a pattern in a new song the degree is
re-read on the new chord in the new scale, and the note is placed in the octave
closest to where the original sat, so register and contour are preserved.
"""

import glob
import os
import re

from .midi_io import read_notes
from .originality import diatonic_index
from .theory import chord_root_pc, scale_notes_ordered

PARTS = ("bass", "arpeggio")
RANGE = {"bass": (24, 60), "arpeggio": (36, 88)}
BASIC = "basic"  # the fixed rule-based pattern the app started with


def _bar_template(notes_in_bar, root_pc, scale):
    root_deg = diatonic_index(root_pc, scale) % 7
    return {
        "root_pc": root_pc,
        "notes": [(round(o * 4) / 4, p, max(0.25, round(d * 4) / 4),
                   (diatonic_index(p, scale) - root_deg) % 7) for o, p, d in notes_in_bar],
    }


def _signature(bars):
    """Key-independent fingerprint, to drop duplicated patterns."""
    sig = []
    for b in bars:
        if not b["notes"]:
            sig.append(())
            continue
        base = b["notes"][0][1]
        sig.append(tuple((o, d, c, (p - base)) for o, p, d, c in b["notes"]))
    return tuple(sig)


def load_patterns(library_dir, metadata_rows, beats_per_bar=4):
    """{part: [pattern]} with pattern = {id, source, bars, notes_per_bar}."""
    meta = {}
    for r in metadata_rows:
        m = re.search(r"(\d+)\s*$", str(r.get("Name", "")))
        if m:
            meta[int(m.group(1))] = r
    out = {p: [] for p in PARTS}
    seen = {p: set() for p in PARTS}
    for folder in sorted(glob.glob(os.path.join(library_dir, "song_*"))):
        m = re.search(r"song_(\d+)$", folder)
        if not m or int(m.group(1)) not in meta:
            continue
        num = int(m.group(1))
        row = meta[num]
        try:
            scale = scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
            roots = [chord_root_pc(c) for c in str(row["chords"]).split("-") if c.strip()]
        except Exception:  # noqa: BLE001
            continue
        if not roots:
            continue
        for part in PARTS:
            files = sorted(glob.glob(os.path.join(folder, f"{part}*.mid")))
            if not files:
                continue
            try:
                notes = read_notes(files[0], top_voice_only=False)
            except Exception:  # noqa: BLE001
                continue
            if not notes:
                continue
            n_bars = max(1, int((max(o + d for o, _, d in notes) - 1e-6) // beats_per_bar) + 1)
            bars = []
            for b in range(n_bars):
                in_bar = [(o - b * beats_per_bar, p, d) for o, p, d in notes
                          if b * beats_per_bar <= o < (b + 1) * beats_per_bar]
                bars.append(_bar_template(in_bar, roots[b % len(roots)], scale))
            if any(not b["notes"] for b in bars):
                continue  # a silent bar would leave holes in the new song
            if part == "bass":
                # one library bass is written an octave up, in the arpeggio's register
                while min(n[1] for b in bars for n in b["notes"]) >= 45:
                    for b in bars:
                        b["notes"] = [(o, p - 12, d, c) for o, p, d, c in b["notes"]]
            sig = _signature(bars)
            if sig in seen[part]:
                continue
            seen[part].add(sig)
            out[part].append({
                "id": f"song_{num:03d}", "source": f"song_{num:03d}", "bars": bars,
                "notes_per_bar": len(notes) / n_bars,
            })
    return out


def _nearest_pitch(pc, target):
    """Pitch with pitch class pc closest to target (ties resolve downward)."""
    base = int(round(target))
    best = None
    for cand in range(base - 11, base + 12):
        if cand % 12 == pc % 12 and (best is None or abs(cand - target) < abs(best - target)):
            best = cand
    return best


def apply_pattern(pattern, part, root_pcs, scale, beats_per_bar=4):
    """Re-voice a pattern on the new chord roots -> [(start, pitch, dur)]."""
    lo, hi = RANGE[part]
    events = []
    bars = pattern["bars"]
    for i, root_pc in enumerate(root_pcs):
        tb = bars[i % len(bars)]
        root_deg = diatonic_index(root_pc, scale) % 7
        shift = (root_pc - tb["root_pc"] + 6) % 12 - 6   # nearest transposition, -6..+5
        pitches = [_nearest_pitch(scale[(root_deg + c) % 7], p + shift) for _, p, _, c in tb["notes"]]
        if min(pitches) < lo:
            pitches = [p + 12 for p in pitches]
        elif max(pitches) > hi:
            pitches = [p - 12 for p in pitches]
        for (o, _, d, _), p in zip(tb["notes"], pitches):
            events.append((i * beats_per_bar + o, p, d))
    return events


def choose(patterns, part, choice, rng):
    """choice: 'random', BASIC or a pattern id. Returns a pattern or None (= basic)."""
    pool = patterns.get(part, [])
    if choice == BASIC or not pool:
        return None
    if choice == "random":
        return pool[int(rng.integers(len(pool)))]
    for p in pool:
        if p["id"] == choice:
            return p
    return None
