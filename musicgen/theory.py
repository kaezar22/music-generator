"""Scales, chords and the rule-based bass / arpeggio / harmony parts.

Same logic as the notebook: chords are built by stacking thirds inside the
song's own scale, and chord progressions are taken from the existing songs
(as scale degrees, tagged by mood) and transposed to the requested scale.
"""

import re

STEP = 0.25  # one 16th note, in beats

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)")
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

IONIAN = [0, 2, 4, 5, 7, 9, 11]
MIXOLYDIAN = [0, 2, 4, 5, 7, 9, 10]
NAT_MINOR = [0, 2, 3, 5, 7, 8, 10]
HARM_MINOR = [0, 2, 3, 5, 7, 8, 11]

# Values offered in the UI, spelled the way the Google Sheet already uses them.
VARIATIONS = {"major": ["natural", "mixolidian"], "minor": ["natural", "harmonic"]}


def key_to_pc(key):
    m = ROOT_RE.match(str(key).strip())
    pc = NOTE_TO_PC[m.group(1).upper()]
    if m.group(2) == "#":
        pc += 1
    elif m.group(2) == "b":
        pc -= 1
    return pc % 12


def scale_notes_ordered(key, mode, variation="natural"):
    root = key_to_pc(key)
    mode = str(mode).strip().lower()
    variation = str(variation).strip().lower()
    if mode == "major":
        intervals = MIXOLYDIAN if ("mixolid" in variation or "mixolyd" in variation) else IONIAN
    elif mode == "minor":
        intervals = HARM_MINOR if "harmonic" in variation else NAT_MINOR
    else:
        intervals = IONIAN
    return [(root + i) % 12 for i in intervals]


chord_root_pc = key_to_pc


def chord_tones(root_pc, scale):
    """Diatonic triad [root, third, fifth] by stacking thirds in the scale."""
    if root_pc in scale:
        i = scale.index(root_pc)
        return [scale[i], scale[(i + 2) % 7], scale[(i + 4) % 7]]
    return [root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12]


def chord_label(root_pc, tones):
    third = (tones[1] - root_pc) % 12
    fifth = (tones[2] - root_pc) % 12
    if third == 3 and fifth == 6:
        q = "dim"
    elif third == 4 and fifth == 8:
        q = "aug"
    elif third == 3:
        q = "m"
    else:
        q = ""
    return NOTE_NAMES[root_pc] + q


# ---------------------------------------------------------------- progressions
def build_progression_library(metadata_rows):
    """[{mood, mode, degrees, source}] from metadata rows (dicts)."""
    lib = []
    for row in metadata_rows:
        try:
            scale = scale_notes_ordered(row["Key"], row["mode"], row.get("variation", "natural"))
            chords = [c.strip() for c in str(row["chords"]).split("-") if c.strip()]
        except Exception:
            continue
        degrees = []
        for label in chords:
            pc = chord_root_pc(label)
            if pc not in scale:
                degrees = None  # borrowed chord -> skip, as in the notebook
                break
            degrees.append(ROMAN[scale.index(pc)])
        if degrees:
            lib.append({
                "mood": str(row.get("mood", "")).strip().lower(),
                "mode": str(row.get("mode", "")).strip().lower(),
                "degrees": degrees,
                "source": str(row.get("Name", "")),
            })
    return lib


def pick_progression(library, mood, mode, rng):
    mood, mode = mood.lower(), mode.lower()
    pool = [p for p in library if p["mood"] == mood and p["mode"] == mode]
    if not pool:
        pool = [p for p in library if p["mood"] == mood]
    if not pool:
        pool = library
    return pool[int(rng.integers(len(pool)))]


def realize_progression(degrees, scale, length_bars):
    """-> (labels, root_pcs) looped to length_bars."""
    labels, roots = [], []
    for d in degrees:
        root = scale[ROMAN.index(d)]
        roots.append(root)
        labels.append(chord_label(root, chord_tones(root, scale)))
    n = len(roots)
    return [labels[i % n] for i in range(length_bars)], [roots[i % n] for i in range(length_bars)]


# ---------------------------------------------------------------- parts
def _voice(root_pc, scale, low=48):
    """Close-position triad starting with the root in [low, low+11]."""
    tones = chord_tones(root_pc, scale)
    root = low + (tones[0] - low) % 12
    third = root + (tones[1] - tones[0]) % 12
    fifth = root + (tones[2] - tones[0]) % 12
    return [root, third, fifth]


def bass_events(root_pcs, beats_per_bar=4, base_pitch=36):
    """One sustained root per bar (notebook template)."""
    return [(i * beats_per_bar, base_pitch + pc, float(beats_per_bar)) for i, pc in enumerate(root_pcs)]


def arpeggio_events(root_pcs, scale, beats_per_bar=4, low=48):
    """8 eighth notes per bar, root-3rd-5th-3rd (notebook template)."""
    pattern = [0, 1, 2, 1, 0, 1, 2, 1]
    ev = []
    for bar, pc in enumerate(root_pcs):
        tones = _voice(pc, scale, low)
        for k, t in enumerate(pattern):
            ev.append((bar * beats_per_bar + k * 0.5, tones[t], 0.5))
    return ev


def harmony_events(root_pcs, scale, beats_per_bar=4, low=48):
    """Block triad sustained for the whole bar, like the existing harmony files."""
    ev = []
    for bar, pc in enumerate(root_pcs):
        for p in _voice(pc, scale, low):
            ev.append((bar * beats_per_bar, p, float(beats_per_bar)))
    return ev


def clip_events(events, total_beats):
    out = []
    for s, p, d in events:
        if s >= total_beats:
            continue
        out.append((s, p, min(d, total_beats - s)))
    return out
