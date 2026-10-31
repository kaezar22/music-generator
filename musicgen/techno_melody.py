"""Melodic techno melodies without a trained model (5 songs are too few for an LSTM).

lead (melody)
    Takes the RHYTHM of one of the user's rhythmic hooks (003, 004, 005…), never
    its pitches. Pitches are new: chord tones on strong beats and long notes,
    scale steps elsewhere, with the template's own share of repeated notes and
    leaps. A motif the length of the template phrase is written once and
    repeated on the following chords (re-voiced by chord degree, like a hook);
    every second phrase rewrites its last bar, and the song ends on the root.

counter-line (melody2)
    By rules: the rhythm of one of the user's slow pad-like lines (001, 002) or
    one note per chord, chord tones with the smallest movement, below the lead,
    preferring 3rds/6ths with it and avoiding unisons and parallel 5ths/8ves.
"""

import os

import numpy as np

from .midi_io import read_notes

BEATS_PER_BAR = 4
HOOK_MIN_NOTES_PER_BAR = 2.5        # above: rhythmic hook (lead); below: slow line (counter)
LEAD_RANGE = (-7, 9)                # semitones around the lead centre
COUNTER_BELOW = 8                   # counter-line centre, semitones under the lead centre


# ------------------------------------------------------------------ templates
def load_templates(library_dir, metadata_rows, columns=("melody", "melody2", "melody3"), read_options=None):
    """Rhythm templates from every melody file of the library.

    -> [{id, source, column, kind, bars, rhythm=[(onset, dur)], repeat_ratio, mean_step}]"""
    out = []
    for row in metadata_rows:
        name = str(row.get("Name", "")).strip()
        digits = "".join(ch for ch in name if ch.isdigit())
        if not digits:
            continue
        folder = os.path.join(library_dir, f"song_{int(digits):03d}")
        for col in columns:
            fname = str(row.get(col, "") or "").strip()
            path = os.path.join(folder, fname)
            if not fname or not os.path.isfile(path):
                continue
            notes = read_notes(path, top_voice_only=True, monophonic=True, **(read_options or {}))
            if len(notes) < 2:
                continue
            total_bars = max(1, int(np.ceil(max(o + d for o, _, d in notes) / BEATS_PER_BAR - 1e-9)))
            phrase = _phrase_bars(notes, total_bars)
            rhythm = [(o, d) for o, _, d in notes if o < phrase * BEATS_PER_BAR]
            rhythm = [(o, min(d, phrase * BEATS_PER_BAR - o)) for o, d in rhythm]
            pitches = [p for _, p, _ in notes]
            steps = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
            density = len(notes) / total_bars
            sid = f"song_{int(digits):03d}"
            out.append({
                "id": sid if col == "melody" else f"{sid}/{col}",
                "source": sid, "column": col,
                "kind": "hook" if density >= HOOK_MIN_NOTES_PER_BAR else "slow",
                "bars": phrase, "rhythm": rhythm,
                "notes_per_bar": len(rhythm) / phrase,
                "repeat_ratio": sum(1 for s in steps if s == 0) / max(1, len(steps)),
                "mean_step": float(np.mean([s for s in steps if s])) if any(steps) else 2.0,
            })
    return out


def _phrase_bars(notes, total_bars):
    """4-bar phrase if the rhythm of bars 1-4 repeats in 5-8, else 8 (or the whole file if shorter)."""
    def rhythm_of(a, b):
        return [(round(o - a * BEATS_PER_BAR, 2), round(d, 2)) for o, _, d in notes
                if a * BEATS_PER_BAR <= o < b * BEATS_PER_BAR]
    if total_bars <= 4:
        return total_bars
    if total_bars >= 8 and rhythm_of(0, 4) == rhythm_of(4, 8):
        return 4
    return min(8, total_bars) if total_bars >= 8 else 4


def choose(templates, kind, choice, rng):
    pool = [t for t in templates if t["kind"] == kind]
    if choice not in (None, "random"):
        for t in templates:
            if t["id"] == choice:
                return t
    return pool[int(rng.integers(len(pool)))] if pool else None


# ------------------------------------------------------------------ helpers
def _candidates(pcs, lo, hi):
    return [p for p in range(lo, hi + 1) if p % 12 in pcs]


def _pick(prev, cands, weights_fn, rng, temperature):
    w = np.array([weights_fn(prev, c) for c in cands], dtype=float)
    w = np.maximum(w, 1e-9) ** (1.0 / max(temperature, 0.05))
    return cands[int(rng.choice(len(cands), p=w / w.sum()))]


def _is_strong(onset, dur):
    pos = onset % BEATS_PER_BAR
    return abs(pos) < 1e-9 or abs(pos - 2) < 1e-9 or dur >= 1.0


def _degree_in(pitch, scale):
    """(scale index, octave) of a pitch; notes outside the scale use the nearest degree."""
    pc = pitch % 12
    for delta in (0, -1, 1, -2, 2):
        if (pc + delta) % 12 in scale:
            return scale.index((pc + delta) % 12), (pitch + delta) // 12
    return 0, pitch // 12


def _revoice(pitch, src_bar, dst_bar, roots, bar_scales):
    """Move a pitch written on one chord to the matching chord degree of another."""
    s_src, s_dst = bar_scales[src_bar], bar_scales[dst_bar]
    deg, _ = _degree_in(pitch, s_src)
    rel = (deg - s_src.index(roots[src_bar])) % 7 if roots[src_bar] in s_src else deg
    root_idx = s_dst.index(roots[dst_bar]) if roots[dst_bar] in s_dst else 0
    pc = s_dst[(root_idx + rel) % 7]
    shift = (roots[dst_bar] - roots[src_bar] + 6) % 12 - 6
    target = pitch + shift
    return min((target + k for k in range(-6, 7) if (target + k) % 12 == pc), key=lambda q: abs(q - target))


# ------------------------------------------------------------------ lead
def make_lead(template, roots, chord_pcs, bar_scales, length_bars, rng, temperature=0.9, center=72):
    """Lead melody on the template's rhythm -> [(start, pitch, dur)]."""
    lo, hi = center + LEAD_RANGE[0], center + LEAD_RANGE[1]
    phrase = template["bars"]
    repeat_w = 0.4 + 2.0 * template["repeat_ratio"]
    step_pref = template["mean_step"]

    def weights(prev, cand):
        if prev is None:
            return 1.0 / (1 + abs(cand - center) / 3)
        jump = abs(cand - prev)
        if jump == 0:
            w = repeat_w
        else:
            w = np.exp(-((jump - step_pref) ** 2) / (2 * 2.5 ** 2)) + (0.6 if jump <= 2 else 0)
        return w * (1.0 / (1 + abs(cand - center) / 6))   # stay near the centre

    def write(bar0, prev, rhythm):
        """Pitches for one pass of the rhythm starting at bar0."""
        out = []
        for o, d in rhythm:
            bar = bar0 + int(o // BEATS_PER_BAR)
            if bar >= length_bars:
                break
            pcs = chord_pcs[bar] if _is_strong(o, d) else set(bar_scales[bar])
            cands = _candidates(pcs, lo, hi) or _candidates(set(bar_scales[bar]), lo, hi)
            prev = _pick(prev, cands, weights, rng, temperature)
            out.append((bar0 * BEATS_PER_BAR + o, prev, d))
        return out, prev

    motif, prev = write(0, None, template["rhythm"])
    events = list(motif)
    for k, bar0 in enumerate(range(phrase, length_bars, phrase)):
        repeat = []
        for start, pitch, dur in motif:
            src_bar = int(start // BEATS_PER_BAR)
            dst_bar = bar0 + src_bar
            if dst_bar >= length_bars:
                break
            repeat.append((bar0 * BEATS_PER_BAR + start, _revoice(pitch, src_bar, dst_bar, roots, bar_scales), dur))
        if k % 2 == 0 and repeat:   # every second phrase: new last bar (answer)
            last = bar0 + phrase - 1
            keep = [e for e in repeat if int(e[0] // BEATS_PER_BAR) < last]
            tail = [(o - (phrase - 1) * BEATS_PER_BAR, d) for o, d in template["rhythm"]
                    if int(o // BEATS_PER_BAR) == phrase - 1]
            new_tail, _ = write(last, keep[-1][1] if keep else None, tail)
            repeat = keep + new_tail
        events.extend(repeat)
    return _end_on_root(events, roots, length_bars, lo, hi)


def _end_on_root(events, roots, length_bars, lo, hi):
    if not events:
        return events
    s, p, d = events[-1]
    root = roots[min(length_bars - 1, int(s // BEATS_PER_BAR))]
    cands = [q for q in range(lo, hi + 1) if q % 12 == root]
    if cands:
        events[-1] = (s, min(cands, key=lambda q: abs(q - p)), d)
    return events


# ------------------------------------------------------------------ counter-line
def make_counter(template, lead, roots, chord_pcs, bar_scales, length_bars, rng, center=72):
    """Counter-line under the lead -> [(start, pitch, dur)]."""
    c_center = center - COUNTER_BELOW
    lo, hi = c_center - 7, c_center + 6
    if template:
        phrase = template["bars"]
        rhythm = [(b0 * BEATS_PER_BAR + o, d) for b0 in range(0, length_bars, phrase)
                  for o, d in template["rhythm"] if b0 * BEATS_PER_BAR + o < length_bars * BEATS_PER_BAR]
        rhythm = _vary_rhythm(rhythm, rng)
    else:  # one note per chord change
        rhythm, b = [], 0
        while b < length_bars:
            e = b + 1
            while e < length_bars and roots[e] == roots[b] and chord_pcs[e] == chord_pcs[b]:
                e += 1
            rhythm.append((b * BEATS_PER_BAR, (e - b) * BEATS_PER_BAR))
            b = e

    def lead_at(t):
        sounding = [p for s, p, d in lead if s <= t < s + d]
        return sounding[-1] if sounding else None

    out, prev, prev_lead = [], None, None
    for start, dur in rhythm:
        bar = int(start // BEATS_PER_BAR)
        cands = _candidates(chord_pcs[bar], lo, hi) or _candidates(set(bar_scales[bar]), lo, hi)
        ld = lead_at(start)
        if ld is not None:
            below = [c for c in cands if c < ld and (ld - c) % 12]
            cands = below or cands   # under the lead and never doubling it, whenever possible
        best, best_score = None, None
        for c in cands:
            score = 0.0
            score -= abs(c - prev) if prev is not None else abs(c - c_center) * 0.5   # smallest movement
            if ld is not None:
                iv = (ld - c) % 12
                if c >= ld or iv == 0:
                    score -= 20                       # stay below, never double the lead
                elif iv in (3, 4, 8, 9):
                    score += 3                        # 3rds and 6ths
                elif iv == 7:
                    score -= 1
                if prev is not None and prev_lead is not None:
                    piv = (prev_lead - prev) % 12
                    same_dir = (ld - prev_lead) * (c - prev) > 0
                    if iv == piv and iv in (0, 7) and same_dir:
                        score -= 10                   # parallel 5ths / octaves
                    if (ld - prev_lead) * (c - prev) < 0:
                        score += 1                    # contrary motion
            score += float(rng.normal(0, 0.3))        # small variety between generations
            if best_score is None or score > best_score:
                best, best_score = c, score
        if best is None:
            continue
        out.append((start, best, dur))
        prev, prev_lead = best, ld
    return out


def _vary_rhythm(rhythm, rng, merge_p=0.35, split_p=0.3):
    """The counter-line's own rhythm: joins neighbours inside a bar and splits
    long notes, so it does not replay the template note for note."""
    out = []
    for o, d in rhythm:
        if out and int(out[-1][0] // BEATS_PER_BAR) == int(o // BEATS_PER_BAR) \
                and abs(out[-1][0] + out[-1][1] - o) < 1e-9 and rng.random() < merge_p:
            out[-1] = (out[-1][0], out[-1][1] + d)
        elif d >= 2 and rng.random() < split_p:
            half = round(d / 2 * 2) / 2
            out += [(o, half), (o + half, d - half)]
        else:
            out.append((o, d))
    return out
