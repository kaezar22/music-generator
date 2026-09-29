"""Scale + mood + length + temperature -> the four parts of a new song."""

import io
import zipfile

import numpy as np

from . import theory
from .melody import symbols_to_events
from .midi_io import events_to_midi_bytes

BPM = 60
BEATS_PER_BAR = 4
TIME_SIG = "4/4"
ENERGY = 0.2
STYLE = "ambient"
PARTS = ("melody", "arpeggio", "bass", "harmony")


def generate_song(model, progression_library, key, mode, variation, mood, length_bars,
                  temperature, seed=None):
    rng = np.random.default_rng(seed)
    scale = theory.scale_notes_ordered(key, mode, variation)
    total_beats = length_bars * BEATS_PER_BAR

    prog = theory.pick_progression(progression_library, mood, mode, rng)
    labels, roots = theory.realize_progression(prog["degrees"], scale, length_bars)
    n_loop = len(prog["degrees"])

    symbols = model.generate_symbols(length_bars * BEATS_PER_BAR * 4, temperature, rng)
    melody = theory.clip_events(symbols_to_events(symbols, scale), total_beats)

    return {
        "key": key, "mode": mode, "variation": variation, "mood": mood,
        "length_bars": length_bars, "temperature": temperature, "seed": seed,
        "scale": scale, "total_beats": total_beats,
        "chords": labels[:n_loop],  # one loop of the progression (sheet format)
        "chords_full": labels,
        "template_source": prog["source"], "template_degrees": prog["degrees"],
        "events": {
            "melody": melody,
            "arpeggio": theory.arpeggio_events(roots, scale, BEATS_PER_BAR),
            "bass": theory.bass_events(roots, BEATS_PER_BAR),
            "harmony": theory.harmony_events(roots, scale, BEATS_PER_BAR),
        },
    }


def midi_files(song, tag):
    """{filename: bytes} named like the existing library: melody_021.mid, ..."""
    return {f"{part}_{tag}.mid": events_to_midi_bytes(song["events"][part], BPM) for part in PARTS}


def song_zip(song, tag):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, data in midi_files(song, tag).items():
            z.writestr(f"song_{tag}/{fname}", data)
    return buf.getvalue()


def sheet_record(song, number, tag):
    return {
        "Name": f"song {number}",
        "Key": song["key"].lower(),
        "mode": song["mode"],
        "variation": song["variation"],
        "bpm": BPM,
        "TimSig": TIME_SIG,
        "mood": song["mood"],
        "energy": ENERGY,
        "style": STYLE,
        "chords": "-".join(song["chords"]),
        "melody": f"melody_{tag}.mid",
        "harmony": f"harmony_{tag}.mid",
        "arpeggio": f"arpeggio_{tag}.mid",
        "bass": f"bass_{tag}.mid",
    }
