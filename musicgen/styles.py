"""Every difference between musical styles lives here.

The pipeline and the app read these fields instead of asking "is this lo-fi?",
so a new style is a new entry, not new branches in the code.
"""

import os

from . import patterns
from .moods import MOOD_GROUPS, TECHNO_MOOD_GROUPS
from .theory import NOTE_NAMES

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Parts every style can use, in display order. Labels/colours are shared.
PART_LABELS = {"melody": "Melodía", "melody2": "Contramelodía", "arpeggio": "Arpegio",
               "arpeggio2": "Arpegio 2", "bass": "Bajo", "harmony": "Armonía"}
PART_COLORS = {"melody": "#2a78d6", "melody2": "#4a3aa7", "arpeggio": "#eb6834",
               "arpeggio2": "#e87ba4", "bass": "#1baf7a", "harmony": "#a3a29c"}

STYLES = {
    "ambient": {
        "label": "Ambient",
        "enabled": True,
        # --- data
        "library": os.path.join(APP_DIR, "library"),
        "models": os.path.join(APP_DIR, "models"),
        "gid_secret": "worksheet_gid", "gid_default": 1183065346,
        "melody_tab": "melodies",
        "folder_hint": "AI_MUSIC_Project",
        "compare_with": [],                      # other styles' melodies checked for copies
        # --- parts
        "parts": ("melody", "arpeggio", "bass", "harmony"),
        "pattern_parts": patterns.PARTS,         # parts learned from the library as patterns
        "pattern_ranges": patterns.RANGE,
        "pattern_options": {},                   # extra load_patterns options (columns, held bars…)
        "split_held": False,                     # held notes re-voiced where the chord changes
        "export_loops": False,                   # accompaniment exported as its loop (not full length)
        # --- harmony
        "strict_scale": False,                   # unknown variations fall back silently (legacy)
        "chords": "triads",                      # triads | extended | original
        "harmony": "block_triads",               # block_triads | patterns
        # --- melody
        "melody_center": None,                   # MIDI pitch the melody is centred on (None = as decoded)
        "harmonic_sevenths": False,              # harmonic minor: raised 7th only on dominants
        "snap_melody": False,                    # move rare chromatic melody notes into scale/chord
        # --- tempo / energy / mood
        "bpm_range": None, "fixed_bpm": 60,
        "energy_range": None, "fixed_energy": 0.2,
        "mood_groups": None,                     # {group: [moods]}: progressions chosen by group
        "group_defaults": {},                    # group -> (bpm, energy) suggested in the UI
        # --- naming / Sheet
        "sheet_style": "ambient",
        "file_prefix": "",
        "name_format": "song {n}",               # Name column in the Sheet
        # --- UI
        "keys": NOTE_NAMES,
        "variations": {"major": ["natural", "mixolidian"], "minor": ["natural", "harmonic"]},
        "moods": ["quiet", "melancolic", "cinematic", "optimistic", "flamenco"],
        "default_mood": "quiet",
        "default_volumes": {"melody": 1.0, "arpeggio": 0.45, "bass": 0.6, "harmony": 0.0},
        "fixed_caption": "Fijos: 60 bpm · 4/4 · energy 0.2 · ambient",
    },
    "lofi": {
        "label": "Lo-fi",
        "enabled": True,
        "library": os.path.join(APP_DIR, "library_lofi"),
        "models": os.path.join(APP_DIR, "models", "lofi"),
        "gid_secret": "lofi_worksheet_gid", "gid_default": 913708363,
        "melody_tab": "melodies lofi",
        "folder_hint": "lofi_songs",
        "compare_with": ["ambient"],
        "parts": ("melody", "arpeggio", "bass", "harmony"),
        "pattern_parts": patterns.LOFI_PARTS,
        "pattern_ranges": patterns.LOFI_RANGE,
        "pattern_options": {},
        "split_held": False,
        "export_loops": True,
        "strict_scale": True,
        "chords": "extended",
        "harmony": "patterns",
        "melody_center": 72,
        "harmonic_sevenths": True,
        "snap_melody": True,
        "bpm_range": (70, 90), "fixed_bpm": None,
        "energy_range": (0.25, 0.6), "fixed_energy": None,
        "mood_groups": MOOD_GROUPS,
        "group_defaults": {"calma": (76, 0.35), "melancolico": (72, 0.35), "luminoso": (84, 0.5)},
        "sheet_style": "lofi",
        "file_prefix": "lofi_",
        "name_format": "song{n:03d}",
        "keys": ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        "variations": {"major": ["natural", "lydian", "mixolydian"], "minor": ["natural", "dorian", "harmonic"]},
        "moods": [m for g in MOOD_GROUPS.values() for m in g
                  if m not in ("quiet", "melancolic", "cinematic", "flamenco")],
        "default_mood": "chill",
        "default_volumes": {"melody": 1.0, "arpeggio": 0.4, "bass": 0.6, "harmony": 0.45},
        "fixed_caption": "4/4 · style lofi · MIDI rectos (el swing lo pones en tu DAW)",
    },
    "worakls": {
        "label": "Melodic techno",
        "enabled": True,
        "library": os.path.join(APP_DIR, "library_techno"),
        "models": os.path.join(APP_DIR, "models", "techno"),   # LSTM trained with tools/train_techno.py
        "melody_source": "hybrid",               # lstm | templates | hybrid (model or rhythm template)
        "melody_columns": ["melody", "melody2", "melody3"],  # rhythm templates + originality library
        "gid_secret": "techno_worksheet_gid", "gid_default": 297000043,
        "melody_tab": "melodies techno",
        "folder_hint": "melodic_ideas",
        "compare_with": [],
        "parts": ("melody", "melody2", "arpeggio", "arpeggio2", "bass", "harmony"),
        "pattern_parts": ("bass", "arpeggio", "harmony"),
        "pattern_ranges": patterns.TECHNO_RANGE,
        "pattern_options": {
            # files come from the Sheet columns, not from their names
            "columns": {"bass": ["bass", "bass2"], "arpeggio": ["arpeggio", "arpeggio2"],
                        "harmony": ["harmony", "harmony2"]},
            "read_options": {"min_dur": 0.125, "quantize": 0.25},  # live takes: drop re-triggers, 1/16 grid
            "allow_held_bars": True,             # pads held over 2 bars
            "chord_aware": True,                 # v minor and V major in the same song
            "keep_chromatic": ("bass", "arpeggio"),  # G# under A (001 bass), C# over F (002 arp2): on purpose
            # takes over another progression/key are left out (009 bass2/harmony2, 010 harmony)
            "min_chord_fit": {"bass": 0.7, "harmony": 0.7},
        },
        "split_held": True,
        "export_loops": True,
        "strict_scale": True,
        "chords": "original",                    # keep the user's v / V
        "harmony": "patterns",
        "melody_center": 72,
        "harmonic_sevenths": False,              # the chords' own scales already decide the 7th
        "snap_melody": True,
        "bpm_range": (120, 140), "fixed_bpm": None,   # 140: the DnB-flavoured songs
        "energy_range": (0.6, 0.95), "fixed_energy": None,
        "mood_groups": TECHNO_MOOD_GROUPS,
        "group_defaults": {"oscuro": (126, 0.8), "emotivo": (124, 0.7), "euforico": (128, 0.85)},
        "sheet_style": "worakls",               # used only when no sub-style is chosen
        "substyles": True,                       # the Sheet's `style` column holds top, balad, DnB…
        "default_substyle": "top",
        "file_prefix": "techno_",
        "name_format": "song_{n:03d}",
        "keys": ["C", "C#", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"],
        "variations": {"major": ["natural"], "minor": ["natural", "harmonic", "dorian", "phrygian"]},
        "moods": [m for g in TECHNO_MOOD_GROUPS.values() for m in g],
        "default_mood": "hypnotic",
        "default_volumes": {"melody": 1.0, "melody2": 0.6, "arpeggio": 0.5, "arpeggio2": 0.4,
                            "bass": 0.7, "harmony": 0.45},
        "fixed_caption": "4/4 · MIDI rectos (cuantizados a 1/16, sin swing: el groove lo pones en tu DAW)",
    },
}


def get(style):
    return STYLES[style]


def melody_parts(cfg):
    """Melodic lines of a style (checked for originality): melody, melody2, …"""
    return [p for p in cfg["parts"] if p.startswith("melody")]
