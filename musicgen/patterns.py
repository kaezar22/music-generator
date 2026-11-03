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
from .theory import chord_root_pc, chord_scale, parse_chord, scale_for_chord, scale_notes_ordered

PARTS = ("bass", "arpeggio")
LOFI_PARTS = ("bass", "arpeggio", "harmony")  # lo-fi also learns the comping rhythm
RANGE = {"bass": (24, 60), "arpeggio": (36, 88), "harmony": (40, 84)}
LOFI_RANGE = {"bass": (24, 60), "arpeggio": (36, 100), "harmony": (40, 86)}
TECHNO_RANGE = {"bass": (24, 60), "arpeggio": (36, 100), "harmony": (24, 86)}  # techno pads may carry their own root low
BASIC = "basic"  # the fixed rule-based pattern the app started with


def _bar_template(notes_in_bar, root_pc, scale, keep_chromatic=False):
    """Notes of one bar as (onset, pitch, dur, chord degree[, semitone offset]).

    With keep_chromatic a note outside the scale keeps its offset from the
    nearest degree (the G# under A in a bass line), instead of being snapped."""
    root_deg = diatonic_index(root_pc, scale) % 7
    notes = []
    for o, p, d in notes_in_bar:
        deg = diatonic_index(p, scale)
        n = (round(o * 4) / 4, p, max(0.25, round(d * 4) / 4), (deg - root_deg) % 7)
        if keep_chromatic:
            n = n + (((p % 12) - scale[deg % 7] + 6) % 12 - 6,)
        notes.append(n)
    return {"root_pc": root_pc, "scale": list(scale), "notes": notes}


def _signature(bars):
    """Key-independent fingerprint, to drop duplicated patterns."""
    sig = []
    for b in bars:
        if not b["notes"]:
            sig.append(())
            continue
        base = b["notes"][0][1]
        sig.append(tuple((n[0], n[2], n[3], n[1] - base) + tuple(n[4:]) for n in b["notes"]))
    return tuple(sig)


def _files_from_columns(folder, row, columns):
    """[(column, path)] named in the song's Sheet/CSV row (cells may hold several
    files separated by ';' in older CSVs)."""
    out = []
    for col in columns:
        for name in str(row.get(col, "") or "").replace(",", ";").split(";"):
            name = name.strip()
            if name and os.path.isfile(os.path.join(folder, name)):
                out.append((col, os.path.join(folder, name)))
    return out


def load_patterns(library_dir, metadata_rows, beats_per_bar=4, parts=PARTS, columns=None,
                  read_options=None, allow_held_bars=False, chord_aware=False, keep_chromatic=(),
                  min_chord_fit=None, dropped=None):
    """{part: [pattern]} with pattern = {id, source, column, bars, notes_per_bar}.

    Default behaviour (ambient, lo-fi): one file per part found by name
    ({part}*.mid or lofi_{part}*.mid), patterns with a bar where no note starts
    are dropped. Options used by techno:
      columns          {part: [column, …]}: files come from the metadata row
                       (e.g. arpeggio <- arpeggio + arpeggio2); every file is a pattern
      read_options     extra arguments for read_notes (min_dur, quantize…)
      allow_held_bars  keep patterns whose bars are covered by a held note (pads)
      chord_aware      read degrees against each bar's chord (v minor and V major
                       in the same song) instead of the plain scale
      keep_chromatic   parts that keep their notes outside the scale (bass)
      min_chord_fit    {part: share}: drop a file whose notes (weighted by length)
                       are chord tones of the song's chords less than this share,
                       i.e. a take written over another progression or key
      dropped          optional list that receives (id, reason) of dropped files
    """
    meta = {}
    for r in metadata_rows:
        m = re.search(r"(\d+)\s*$", str(r.get("Name", "")))
        if m:
            meta[int(m.group(1))] = r
    out = {p: [] for p in parts}
    seen = {p: set() for p in parts}
    for folder in sorted(glob.glob(os.path.join(library_dir, "song_*"))):
        m = re.search(r"song_(\d+)$", folder)
        if not m or int(m.group(1)) not in meta:
            continue
        num = int(m.group(1))
        row = meta[num]
        try:
            scale = scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
            labels = [c.strip() for c in str(row["chords"]).split("-") if c.strip()]
            roots = [chord_root_pc(c) for c in labels]
            bar_scales = None
            if chord_aware:
                base = chord_scale(scale, row.get("variation", "natural"), 0)
                bar_scales = [scale_for_chord(base, parse_chord(c)["pcs"]) for c in labels]
        except Exception:  # noqa: BLE001
            continue
        if not roots:
            continue
        for part in parts:
            if columns:
                sources = _files_from_columns(folder, row, columns.get(part, [part]))
            else:
                files = sorted(glob.glob(os.path.join(folder, f"{part}*.mid"))
                               + glob.glob(os.path.join(folder, f"lofi_{part}*.mid")))
                sources = [(part, files[0])] if files else []
            for col, path in sources:
                try:
                    notes = read_notes(path, top_voice_only=False, **(read_options or {}))
                except Exception:  # noqa: BLE001
                    continue
                if not notes:
                    continue
                if min_chord_fit and part in min_chord_fit:
                    fit = _chord_fit(notes, labels, beats_per_bar)
                    if fit < min_chord_fit[part]:
                        if dropped is not None:
                            dropped.append((f"song_{num:03d}/{col}", f"{fit:.0%} notas del acorde"))
                        continue
                n_bars = max(1, int((max(o + d for o, _, d in notes) - 1e-6) // beats_per_bar) + 1)
                if allow_held_bars and n_bars % len(roots) and n_bars > len(roots):
                    # a pad ringing past the loop: count bars up to the last note START
                    by_onset = max(1, int(max(o for o, _, _ in notes) // beats_per_bar) + 1)
                    if by_onset % len(roots) == 0:
                        n_bars = by_onset
                bars = []
                for b in range(n_bars):
                    in_bar = [(o - b * beats_per_bar, p, d) for o, p, d in notes
                              if b * beats_per_bar <= o < (b + 1) * beats_per_bar]
                    sc = bar_scales[b % len(bar_scales)] if bar_scales else scale
                    bars.append(_bar_template(in_bar, roots[b % len(roots)], sc, part in keep_chromatic))
                if any(not b["notes"] for b in bars):
                    if not allow_held_bars or not _covered(notes, n_bars, beats_per_bar):
                        continue  # a silent bar would leave holes in the new song
                if part == "bass":
                    # one library bass is written an octave up, in the arpeggio's register
                    while min(n[1] for b in bars for n in b["notes"]) >= 45:
                        for b in bars:
                            b["notes"] = [(n[0], n[1] - 12) + tuple(n[2:]) for n in b["notes"]]
                sig = _signature(bars)
                if sig in seen[part]:
                    continue
                seen[part].add(sig)
                pid = f"song_{num:03d}" if col == part else f"song_{num:03d}/{col}"
                out[part].append({
                    "id": pid, "source": f"song_{num:03d}", "column": col, "bars": bars,
                    "notes_per_bar": len(notes) / n_bars,
                })
    return out


def _chord_fit(notes, labels, beats_per_bar=4):
    """Share of the notes' length that falls on tones of each bar's chord."""
    pcs = [parse_chord(c)["pcs"] for c in labels]
    total = fit = 0.0
    for o, p, d in notes:
        total += d
        fit += d * (p % 12 in pcs[int(o // beats_per_bar) % len(pcs)])
    return fit / total if total else 1.0


def _covered(notes, n_bars, beats_per_bar):
    """True if every bar has a note starting in it or ringing through it."""
    for b in range(n_bars):
        lo, hi = b * beats_per_bar, (b + 1) * beats_per_bar
        if not any(o < hi and o + d > lo for o, _, d in notes):
            return False
    return True


def _nearest_pitch(pc, target):
    """Pitch with pitch class pc closest to target (ties resolve downward)."""
    base = int(round(target))
    best = None
    for cand in range(base - 11, base + 12):
        if cand % 12 == pc % 12 and (best is None or abs(cand - target) < abs(best - target)):
            best = cand
    return best


def apply_pattern(pattern, part, root_pcs, scale, beats_per_bar=4, ranges=None, bar_scales=None,
                  split_held=False):
    """Re-voice a pattern on the new chord roots -> [(start, pitch, dur)].

    bar_scales: optional scale per bar (harmonic minor uses the raised 7th only
    on the dominant chords). split_held: a note held past its bar is cut where
    the chord changes and continues on the new chord's matching degree (pads)."""
    lo, hi = (ranges or RANGE)[part]
    events = []
    bars = pattern["bars"]
    n_total = len(root_pcs)
    for i, root_pc in enumerate(root_pcs):
        if bar_scales:
            scale = bar_scales[i]
        tb = bars[i % len(bars)]
        root_deg = diatonic_index(root_pc, scale) % 7
        shift = (root_pc - tb["root_pc"] + 6) % 12 - 6   # nearest transposition, -6..+5
        pitches = [_nearest_pitch((scale[(root_deg + n[3]) % 7] + (n[4] if len(n) > 4 else 0)) % 12,
                                  n[1] + shift) for n in tb["notes"]]
        if pitches and min(pitches) < lo:
            pitches = [p + 12 for p in pitches]
        elif pitches and max(pitches) > hi:
            pitches = [p - 12 for p in pitches]
        for n, p in zip(tb["notes"], pitches):
            o, d = n[0], n[2]
            if not split_held:
                events.append((i * beats_per_bar + o, p, d))
                continue
            events.extend(_split_held(i, o, d, p, n, root_pcs, bar_scales, scale, beats_per_bar, n_total))
    return events


def _split_held(i, o, d, p, n, root_pcs, bar_scales, scale, bpb, n_total):
    """Segments of a held note, re-voiced on every chord it crosses."""
    start, end = i * bpb + o, i * bpb + o + d
    segs, bar, pitch = [], i, p
    while start < end - 1e-9:
        seg_end = min(end, (bar + 1) * bpb)
        if segs and segs[-1][1] == pitch:  # same chord: keep the note sounding
            segs[-1] = (segs[-1][0], pitch, seg_end - segs[-1][0])
        else:
            segs.append((start, pitch, seg_end - start))
        start, bar = seg_end, bar + 1
        if bar >= n_total or start >= end - 1e-9:
            break
        sc = bar_scales[bar] if bar_scales else scale
        deg = (diatonic_index(root_pcs[bar], sc) + n[3]) % 7
        pitch = _nearest_pitch((sc[deg] + (n[4] if len(n) > 4 else 0)) % 12, pitch)
    return segs


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
