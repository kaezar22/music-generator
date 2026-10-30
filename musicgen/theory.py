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
LYDIAN = [0, 2, 4, 6, 7, 9, 11]
NAT_MINOR = [0, 2, 3, 5, 7, 8, 10]
HARM_MINOR = [0, 2, 3, 5, 7, 8, 11]
DORIAN = [0, 2, 3, 5, 7, 9, 10]

# Values offered in the UI, spelled the way the Google Sheet already uses them.
VARIATIONS = {"major": ["natural", "mixolidian"], "minor": ["natural", "harmonic"]}

# Every (mode, variation) the datasets use. Old ambient sheet spells "mixolidian".
SCALES = {
    ("major", "natural"): IONIAN,
    ("major", "ionian"): IONIAN,
    ("major", "lydian"): LYDIAN,
    ("major", "mixolydian"): MIXOLYDIAN,
    ("major", "mixolidian"): MIXOLYDIAN,
    ("minor", "natural"): NAT_MINOR,
    ("minor", "aeolian"): NAT_MINOR,
    ("minor", "harmonic"): HARM_MINOR,
    ("minor", "dorian"): DORIAN,
}

KEY_RE = re.compile(r"^([a-g])([#b]?)$")


def parse_key(key):
    """Strict key parser for Sheet values: 'c', 'f#', 'eb', 'bb', 'b' (= B natural).

    The accidental is only read from a SECOND character, so a lone 'b' is B,
    never a flat. Raises ValueError on anything else."""
    k = str(key).strip().lower()
    m = KEY_RE.match(k)
    if not m:
        raise ValueError(f"key no reconocida: {key!r}")
    pc = NOTE_TO_PC[m.group(1).upper()]
    pc += {"#": 1, "b": -1, "": 0}[m.group(2)]
    return pc % 12


def parse_scale(key, mode, variation):
    """Strict: -> ordered scale pitch classes. Raises ValueError for unknown or
    incompatible values (e.g. 'dorian' with mode 'major')."""
    mode = str(mode).strip().lower()
    variation = str(variation).strip().lower()
    if (mode, variation) not in SCALES:
        raise ValueError(f"combinación mode/variation no reconocida: {mode!r}/{variation!r}")
    root = parse_key(key)
    return [(root + i) % 12 for i in SCALES[(mode, variation)]]


def key_to_pc(key):
    m = ROOT_RE.match(str(key).strip())
    pc = NOTE_TO_PC[m.group(1).upper()]
    if m.group(2) == "#":
        pc += 1
    elif m.group(2) == "b":
        pc -= 1
    return pc % 12


def scale_notes_ordered(key, mode, variation="natural"):
    """Lenient version used by the app: unknown variations fall back to the
    plain major / natural minor scale."""
    mode = str(mode).strip().lower()
    variation = str(variation).strip().lower()
    try:
        return parse_scale(key, mode, variation)
    except ValueError:
        pass
    root = key_to_pc(key)
    if mode == "major":
        intervals = MIXOLYDIAN if ("mixolid" in variation or "mixolyd" in variation) else IONIAN
    elif mode == "minor":
        intervals = HARM_MINOR if "harmonic" in variation else NAT_MINOR
    else:
        intervals = IONIAN
    return [(root + i) % 12 for i in intervals]


chord_root_pc = key_to_pc

# ---------------------------------------------------------------- chord labels
# Intervals (semitones above the root) for the chord types in the datasets.
CHORD_QUALITIES = {
    "": (0, 4, 7), "maj": (0, 4, 7), "m": (0, 3, 7), "min": (0, 3, 7),
    "dim": (0, 3, 6), "aug": (0, 4, 8), "+": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "sus": (0, 5, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9), "add9": (0, 4, 7, 14), "madd9": (0, 3, 7, 14),
    "maj7": (0, 4, 7, 11), "maj9": (0, 4, 7, 11, 14), "maj13": (0, 4, 7, 11, 14, 21),
    "maj7#11": (0, 4, 7, 11, 18),
    "m7": (0, 3, 7, 10), "m9": (0, 3, 7, 10, 14), "m11": (0, 3, 7, 10, 14, 17),
    "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "mmaj7": (0, 3, 7, 11),
    "7": (0, 4, 7, 10), "9": (0, 4, 7, 10, 14), "11": (0, 4, 7, 10, 14, 17),
    "13": (0, 4, 7, 10, 14, 21), "7b9": (0, 4, 7, 10, 13), "7#9": (0, 4, 7, 10, 15),
    "7sus4": (0, 5, 7, 10), "9sus4": (0, 5, 7, 10, 14),
    "m13": (0, 3, 7, 10, 14, 21), "maj9#11": (0, 4, 7, 11, 14, 18), "9#11": (0, 4, 7, 10, 14, 18),
    "m9b5": (0, 3, 6, 10, 14), "m11b5": (0, 3, 6, 10, 14, 17), "7b9b13": (0, 4, 7, 10, 13, 20),
    "7b13": (0, 4, 7, 10, 20), "m7b9": (0, 3, 7, 10, 13), "m11b9": (0, 3, 7, 10, 13, 17),
    "m7b9b13": (0, 3, 7, 10, 13, 20), "maj7#5": (0, 4, 8, 11), "mmaj9": (0, 3, 7, 11, 14),
    "m7b5b9": (0, 3, 6, 10, 13), "m11b5b9": (0, 3, 6, 10, 13, 17), "m7b5b9b13": (0, 3, 6, 10, 13, 20),
    "m9b13": (0, 3, 7, 10, 14, 20), "m11b13": (0, 3, 7, 10, 14, 17, 20), "maj9#5": (0, 4, 8, 11, 14),
    "9b13": (0, 4, 7, 10, 14, 20), "m9b5b13": (0, 3, 6, 10, 14, 20),
}
CHORD_RE = re.compile(r"^([A-G])([#b]?)([^/]*)(?:/([A-G])([#b]?))?$")


def parse_chord(label):
    """'C#m7b5', 'Bbmaj9', 'Cm9/G' -> {root_pc, bass_pc, quality, pcs, third_pc}.

    Raises ValueError if the label or chord type is not recognised."""
    lab = str(label).strip()
    m = CHORD_RE.match(lab)
    if not m:
        raise ValueError(f"acorde no reconocido: {label!r}")
    root = (NOTE_TO_PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    quality = m.group(3)
    if quality not in CHORD_QUALITIES:
        raise ValueError(f"tipo de acorde no reconocido: {quality!r} en {label!r}")
    intervals = CHORD_QUALITIES[quality]
    bass = root
    if m.group(4):
        bass = (NOTE_TO_PC[m.group(4)] + {"#": 1, "b": -1, "": 0}[m.group(5)]) % 12
    pcs = {(root + i) % 12 for i in intervals} | {bass}
    third = next(((root + i) % 12 for i in intervals if i in (3, 4)), None)
    return {"label": lab, "root_pc": root, "bass_pc": bass, "quality": quality,
            "pcs": pcs, "third_pc": third}


def parse_chords(text):
    """'Dm9-G7-Cmaj7' -> list of parsed chords (cells are strip()-ed)."""
    return [parse_chord(c) for c in str(text).strip().split("-") if c.strip()]



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
        degrees, sizes = [], []
        for label in chords:
            try:
                sizes.append(chord_size(parse_chord(label)["quality"]))
            except ValueError:
                sizes.append("seventh")
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
                "sizes": sizes,
                "variation": str(row.get("variation", "")).strip().lower(),
                "source": str(row.get("Name", "")),
            })
    return lib


def pick_progression(library, mood, mode, rng, group_of=None):
    """Template with the same mood and mode if possible. With group_of (a function
    mood -> group) moods are compared by group, e.g. 'nostalgic' ~ 'melancholic'."""
    mood, mode = mood.lower(), mode.lower()
    key = group_of or (lambda m: m)
    target = key(mood)
    pool = [p for p in library if key(p["mood"]) == target and p["mode"] == mode]
    if not pool:
        pool = [p for p in library if key(p["mood"]) == target]
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


# ---------------------------------------------------------------- extended chords (lo-fi)
# Scale steps stacked above the root for each "size" of chord. A template chord
# keeps its size (7th, 9th, 13th…) but its quality is rebuilt inside the new
# scale, so e.g. a 9th chord on degree V becomes 7b9 in harmonic minor and 9 in
# mixolydian.
STACKS = {
    "triad": (0, 2, 4),
    "six": (0, 2, 4, 5),
    "seventh": (0, 2, 4, 6),
    "ninth": (0, 2, 4, 6, 1),
    "eleventh": (0, 2, 4, 6, 1, 3),
    "thirteenth": (0, 2, 4, 6, 1, 5),
}
_SIZE_OF = {
    "": "triad", "maj": "triad", "m": "triad", "min": "triad", "dim": "triad", "aug": "triad", "+": "triad",
    "sus2": "triad", "sus4": "triad", "sus": "triad",
    "6": "six", "m6": "six", "add9": "triad", "madd9": "triad",
    "maj7": "seventh", "m7": "seventh", "7": "seventh", "m7b5": "seventh", "dim7": "seventh",
    "mmaj7": "seventh", "7sus4": "seventh",
    "maj9": "ninth", "m9": "ninth", "9": "ninth", "7b9": "ninth", "7#9": "ninth", "9sus4": "ninth",
    "maj7#11": "eleventh", "m11": "eleventh", "11": "eleventh",
    "13": "thirteenth", "maj13": "thirteenth", "m13": "thirteenth", "7b9b13": "thirteenth",
    "m7b9b13": "thirteenth", "maj9#11": "eleventh", "9#11": "eleventh", "m11b5": "eleventh",
    "m11b9": "eleventh", "m9b5": "ninth", "m7b9": "ninth", "mmaj9": "ninth", "7b13": "seventh",
    "maj7#5": "seventh", "m7b5b9": "ninth", "m11b5b9": "eleventh", "m7b5b9b13": "thirteenth",
    "m9b13": "thirteenth", "m11b13": "thirteenth", "9b13": "thirteenth", "m9b5b13": "thirteenth",
    "maj9#5": "ninth",
}
_FALLBACK = {"thirteenth": "ninth", "eleventh": "ninth", "ninth": "seventh", "six": "triad",
             "seventh": "triad", "triad": None}
# when several names fit the same notes, prefer these (in order)
_NAME_ORDER = ["maj7", "m7", "7", "m7b5", "maj9", "m9", "9", "7b9", "7#9", "13", "maj13", "m13", "m11", "11",
               "maj7#11", "maj9#11", "9#11", "m9b5", "m11b5", "7b9b13", "7b13", "m7b9", "m11b9", "m7b9b13",
               "m7b5b9", "m11b5b9", "m7b5b9b13", "m9b13", "m11b13", "9b13", "m9b5b13",
               "6", "m6", "", "m", "dim", "aug", "mmaj7", "mmaj9", "maj7#5", "maj9#5", "dim7"]


NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def uses_flats(scale):
    """True if the scale is spelled with flats: the spelling (sharps or flats)
    that gives each of the 7 letters exactly once wins; ties go to sharps."""
    sharp_letters = {NOTE_NAMES[pc][0] for pc in scale}
    flat_letters = {NOTE_NAMES_FLAT[pc][0] for pc in scale}
    return len(flat_letters) > len(sharp_letters)


def pc_name(pc, flats=False):
    return (NOTE_NAMES_FLAT if flats else NOTE_NAMES)[pc % 12]


def chord_size(quality):
    return _SIZE_OF.get(quality, "seventh")


def stacked_chord(root_pc, scale, size, flats=False, flat_scale=None):
    """Chord built by stacking thirds inside the scale -> (label, pcs).

    If the stacked notes have no conventional name (e.g. maj9#5 on III of
    harmonic minor) the size is reduced until one fits. With flat spelling, a
    root outside flat_scale (e.g. the raised 7th of harmonic minor) keeps its
    sharp name: F#dim7 in G minor, not Gbdim7."""
    name = pc_name(root_pc, flats and (flat_scale is None or root_pc in flat_scale))
    if root_pc not in scale:
        return name, {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}
    i = scale.index(root_pc)
    while size:
        ivs = set()
        for step in STACKS[size]:
            pc = scale[(i + step) % 7]
            iv = (pc - root_pc) % 12
            if step in (1, 3) or (step == 5 and size != "six"):  # 9th, 11th, 13th: an octave up
                iv += 12
            ivs.add(iv)
        for q in _NAME_ORDER:
            if set(CHORD_QUALITIES[q]) == ivs:
                return name + q, {(root_pc + v) % 12 for v in ivs}
        size = _FALLBACK[size]
    return name, {root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12}


def chord_scale(scale, variation, degree_index):
    """Scale used to build the chord on a degree. In harmonic minor the raised
    7th belongs only to the dominant chords (V and vii°); the other chords use
    the natural 7th, as in the hand-made songs."""
    if str(variation).strip().lower() == "harmonic" and degree_index not in (4, 6):
        return scale[:6] + [(scale[6] - 1) % 12]
    return scale


def realize_progression_ext(template, scale, length_bars, variation="natural"):
    """template = progression entry with 'degrees' and 'sizes' -> (labels, roots, pcs) looped."""
    labels, roots, pcs = [], [], []
    natural = chord_scale(scale, variation, 0)
    flats = uses_flats(natural)
    for deg, size in zip(template["degrees"], template.get("sizes") or ["seventh"] * len(template["degrees"])):
        d = ROMAN.index(deg)
        root = scale[d]
        lab, chord_pcs = stacked_chord(root, chord_scale(scale, variation, d), size, flats, natural)
        labels.append(lab)
        roots.append(root)
        pcs.append(chord_pcs)
    n = len(roots)
    return ([labels[i % n] for i in range(length_bars)], [roots[i % n] for i in range(length_bars)],
            [pcs[i % n] for i in range(length_bars)])


_SIZE_RANK = ["triad", "six", "seventh", "ninth", "eleventh", "thirteenth"]


def voiced_steps(root_pc, scale, pcs):
    """Scale steps above the root that a voicing plays (0 root, 2 third, 1 ninth…)."""
    if root_pc not in scale:
        return set()
    i = scale.index(root_pc)
    return {(scale.index(pc) - i) % 7 for pc in pcs if pc in scale}


def size_covering(steps):
    """Smallest chord size whose stack contains every scale step in `steps`."""
    for size in _SIZE_RANK:
        if set(steps) <= set(STACKS[size]):
            return size
    return "thirteenth"


def size_for_voicing(root_pc, scale, pcs):
    return size_covering(voiced_steps(root_pc, scale, pcs))


def bigger_size(a, b):
    return a if _SIZE_RANK.index(a) >= _SIZE_RANK.index(b) else b
