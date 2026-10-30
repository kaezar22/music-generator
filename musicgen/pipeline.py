"""Scale + mood + length + temperature -> the four parts of a new song.

Two styles share the same steps:
  ambient  the original app (60 bpm, triads, ambient model)
  lofi     hand-made lo-fi dataset: extended chords, comping patterns,
           its own melody model, bpm/energy chosen per song
"""

import io
import math
import zipfile

import numpy as np

from . import patterns as pat
from . import theory
from .melody import symbols_to_events
from .midi_io import events_to_midi_bytes
from .moods import mood_group

AMBIENT, LOFI = "ambient", "lofi"
BPM = 60
BEATS_PER_BAR = 4
TIME_SIG = "4/4"
ENERGY = 0.2
STYLE = "ambient"
PARTS = ("melody", "arpeggio", "bass", "harmony")
MELODY_CENTER = 72  # C5: lo-fi melodies are moved by whole octaves to sit around it


def center_melody(events, target):
    """Shift the whole melody by octaves so its average pitch (weighted by
    duration) lands within half an octave of target."""
    if not events:
        return events
    total = sum(d for _, _, d in events) or 1.0
    mean = sum(p * d for _, p, d in events) / total
    k = math.floor((target - mean) / 12 + 0.5)
    return [(s, p + 12 * k, d) for s, p, d in events]


def _harmonic_sevenths(melody, scale, bar_scales):
    """Harmonic minor: the raised 7th only over the dominant chords; in the other
    bars the melody uses the natural 7th (same rule as the hand-made songs)."""
    raised = scale[6]
    out = []
    for start, pitch, dur in melody:
        bar = int(start // BEATS_PER_BAR)
        if pitch % 12 == raised and bar < len(bar_scales) and raised not in bar_scales[bar]:
            pitch -= 1
        out.append((start, pitch, dur))
    return out


def _snap_to_harmony(melody, bar_scales, chord_pcs):
    """Lo-fi: a note outside the bar's scale and chord (a rare chromatic note
    from the model) moves a semitone to the nearest allowed note."""
    out = []
    for start, pitch, dur in melody:
        bar = int(start // BEATS_PER_BAR)
        if bar < len(bar_scales):
            allowed = set(bar_scales[bar]) | chord_pcs[bar]
            if pitch % 12 not in allowed:
                pitch = next((pitch + d for d in (-1, 1, -2, 2) if (pitch + d) % 12 in allowed), pitch)
        out.append((start, pitch, dur))
    return out


def _block_chords(chord_pcs, beats_per_bar=4, low=52):
    """Fallback harmony: close voicing of every chord tone, one per bar."""
    ev = []
    for bar, pcs in enumerate(chord_pcs):
        for pc in sorted(pcs, key=lambda x: (x - low) % 12):
            ev.append((bar * beats_per_bar, low + (pc - low) % 12, float(beats_per_bar)))
    return ev


def _names_for_voicings(prog, roots, scale, variation, harmony, labels):
    """Chord names that cover every note the comping pattern actually plays,
    so the Sheet's `chords` column matches harmony.mid (e.g. maj7 -> maj9 when
    the voicing has a 9th)."""
    sizes = prog.get("sizes") or ["seventh"] * len(prog["degrees"])
    natural = theory.chord_scale(scale, variation, 0)
    flats = theory.uses_flats(natural)
    by_bar = {}
    for start, pitch, _ in harmony:
        by_bar.setdefault(int(start // BEATS_PER_BAR), set()).add(pitch % 12)
    out = list(labels)
    for bar, root in enumerate(roots):
        d = scale.index(root)
        bar_scale = theory.chord_scale(scale, variation, d)
        steps = set(theory.STACKS[sizes[bar % len(sizes)]]) | theory.voiced_steps(root, bar_scale, by_bar.get(bar, set()))
        out[bar] = theory.stacked_chord(root, bar_scale, theory.size_covering(steps), flats, natural)[0]
    return out


def generate_song(model, progression_library, key, mode, variation, mood, length_bars,
                  temperature, seed=None, pattern_library=None, bass_choice="random",
                  arp_choice="random", style=AMBIENT, harmony_choice="random", octave=0,
                  bpm=None, energy=None):
    """bass/arp/harmony_choice: "random", "basic" or the id of a library pattern."""
    rng = np.random.default_rng(seed)
    lofi = style == LOFI
    scale = theory.parse_scale(key, mode, variation) if lofi else theory.scale_notes_ordered(key, mode, variation)
    total_beats = length_bars * BEATS_PER_BAR

    if lofi:
        prog = theory.pick_progression(progression_library, mood, mode, rng, group_of=mood_group)
        labels, roots, chord_pcs = theory.realize_progression_ext(prog, scale, length_bars, variation)
        bar_scales = [theory.chord_scale(scale, variation, scale.index(r)) for r in roots]
    else:
        prog = theory.pick_progression(progression_library, mood, mode, rng)
        labels, roots = theory.realize_progression(prog["degrees"], scale, length_bars)
        chord_pcs, bar_scales = None, None
    n_loop = len(prog["degrees"])

    symbols = model.generate_symbols(length_bars * BEATS_PER_BAR * 4, temperature, rng)
    melody = symbols_to_events(symbols, scale, getattr(model, "encoding", "c_octave_v1"))
    if lofi:
        if str(variation).strip().lower() == "harmonic":
            melody = _harmonic_sevenths(melody, scale, bar_scales)
        melody = _snap_to_harmony(melody, bar_scales, chord_pcs)
        melody = center_melody(melody, MELODY_CENTER + 12 * octave)
    melody = theory.clip_events(melody, total_beats)

    pattern_library = pattern_library or {}
    ranges = pat.LOFI_RANGE if lofi else pat.RANGE
    bass_pat = pat.choose(pattern_library, "bass", bass_choice, rng)
    arp_pat = pat.choose(pattern_library, "arpeggio", arp_choice, rng)
    harm_pat = pat.choose(pattern_library, "harmony", harmony_choice, rng) if lofi else None

    def apply(p, part):
        return pat.apply_pattern(p, part, roots, scale, BEATS_PER_BAR, ranges, bar_scales)

    bass = apply(bass_pat, "bass") if bass_pat else theory.bass_events(roots, BEATS_PER_BAR)
    arpeggio = apply(arp_pat, "arpeggio") if arp_pat else theory.arpeggio_events(roots, scale, BEATS_PER_BAR)
    if lofi:
        harmony = apply(harm_pat, "harmony") if harm_pat else _block_chords(chord_pcs, BEATS_PER_BAR)
        labels = _names_for_voicings(prog, roots, scale, variation, harmony, labels)
    else:
        harmony = theory.harmony_events(roots, scale, BEATS_PER_BAR)

    loop_bars = min(length_bars, math.lcm(n_loop, 4)) if lofi else length_bars

    return {
        "style": style,
        "loop_bars": loop_bars,
        "bpm": float(bpm) if (lofi and bpm) else BPM,
        "energy": float(energy) if (lofi and energy is not None) else ENERGY,
        "octave": octave if lofi else 0,
        "bass_choice": bass_choice, "arp_choice": arp_choice, "harmony_choice": harmony_choice,
        "bass_pattern": bass_pat["id"] if bass_pat else pat.BASIC,
        "arp_pattern": arp_pat["id"] if arp_pat else pat.BASIC,
        "harmony_pattern": (harm_pat["id"] if harm_pat else pat.BASIC) if lofi else None,
        "key": key, "mode": mode, "variation": variation, "mood": mood,
        "length_bars": length_bars, "temperature": temperature, "seed": seed,
        "scale": scale, "total_beats": total_beats,
        "chords": labels[:n_loop],  # one loop of the progression (sheet format)
        "chords_full": labels,
        "template_source": prog["source"], "template_degrees": prog["degrees"],
        "events": {
            "melody": melody,
            "arpeggio": theory.clip_events(arpeggio, total_beats),
            "bass": theory.clip_events(bass, total_beats),
            "harmony": theory.clip_events(harmony, total_beats),
        },
    }


def file_prefix(song):
    return "lofi_" if song.get("style") == LOFI else ""


def midi_files(song, tag):
    """{filename: bytes}: melody_021.mid … (ambient) or lofi_melody_021.mid … (lo-fi).

    Lo-fi exports harmony, bass and arpeggio as their repeating loop (4 bars),
    like the hand-made dataset; the melody keeps the full length."""
    pre = file_prefix(song)
    loop_beats = song.get("loop_bars", song["length_bars"]) * BEATS_PER_BAR
    files = {}
    for part in PARTS:
        events = song["events"][part]
        if part != "melody" and song.get("style") == LOFI:
            events = theory.clip_events(events, loop_beats)
        files[f"{pre}{part}_{tag}.mid"] = events_to_midi_bytes(events, song.get("bpm", BPM))
    return files


def song_zip(song, tag):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, data in midi_files(song, tag).items():
            z.writestr(f"song_{tag}/{fname}", data)
    return buf.getvalue()


def sheet_record(song, number, tag):
    pre = file_prefix(song)
    lofi = song.get("style") == LOFI
    bpm = song.get("bpm", BPM)
    return {
        "Name": f"song{number:03d}" if lofi else f"song {number}",
        "Key": song["key"].lower(),
        "mode": song["mode"],
        "variation": song["variation"],
        "bpm": int(bpm) if float(bpm).is_integer() else bpm,
        "TimSig": TIME_SIG,
        "mood": song["mood"],
        "energy": song.get("energy", ENERGY),
        "style": LOFI if lofi else STYLE,
        "chords": "-".join(song["chords"]),
        "melody": f"{pre}melody_{tag}.mid",
        "harmony": f"{pre}harmony_{tag}.mid",
        "arpeggio": f"{pre}arpeggio_{tag}.mid",
        "bass": f"{pre}bass_{tag}.mid",
    }
