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
from . import styles, techno_melody, theory
from .melody import symbols_to_events
from .midi_io import events_to_midi_bytes
from .moods import grouper

AMBIENT, LOFI = "ambient", "lofi"  # nombres de estilo (las diferencias viven en styles.py)
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
                  bpm=None, energy=None, lead_choice="random", arp2_choice="auto", substyle=None):
    """bass/arp/harmony_choice: "random", "basic" or the id of a library pattern.
    Everything that differs between styles comes from musicgen/styles.py."""
    cfg = styles.get(style)
    rng = np.random.default_rng(seed)
    if cfg["strict_scale"]:
        scale = theory.parse_scale(key, mode, variation)
    else:
        scale = theory.scale_notes_ordered(key, mode, variation)
    total_beats = length_bars * BEATS_PER_BAR

    group_of = grouper(cfg["mood_groups"]) if cfg["mood_groups"] else None
    if substyle and cfg.get("substyles"):
        # techno: the user's sub-style (top, balad, DnB…) comes first; its songs give
        # the progression and, when they have one, the patterns and melody rhythms
        progression_library, pattern_library = _prefer_substyle(
            substyle, progression_library, pattern_library, mood, mode, group_of)
    prog = theory.pick_progression(progression_library, mood, mode, rng, group_of=group_of)
    if cfg["chords"] == "extended":
        labels, roots, chord_pcs = theory.realize_progression_ext(prog, scale, length_bars, variation)
        bar_scales = [theory.chord_scale(scale, variation, scale.index(r)) for r in roots]
    elif cfg["chords"] == "original":
        labels, roots, chord_pcs, bar_scales = theory.realize_progression_orig(prog, scale, length_bars, variation, mode)
    else:
        labels, roots = theory.realize_progression(prog["degrees"], scale, length_bars)
        chord_pcs, bar_scales = None, None
    n_loop = len(prog["degrees"])

    centred = cfg["melody_center"] is not None
    templates = (pattern_library or {}).get("melody_templates") or []
    lead_t = None
    source = cfg.get("melody_source", "lstm")
    if source == "hybrid" and lead_choice == "model" and model is not None:
        # melodic techno with its trained model: a motif of `phrase` bars written by
        # the LSTM, fitted to the chords and repeated like a hook
        melody = techno_melody.model_lead(model, scale, roots, chord_pcs, bar_scales, length_bars, rng,
                                          temperature, cfg["melody_center"] + 12 * octave)
        lead_t = {"id": "model"}
    elif source in ("templates", "hybrid"):
        # melodic techno: lead from the user's rhythm templates
        lead_t = techno_melody.choose(templates, "hook", lead_choice, rng)
        melody = techno_melody.make_lead(lead_t, roots, chord_pcs, bar_scales, length_bars, rng,
                                         temperature, cfg["melody_center"] + 12 * octave)
    else:
        symbols = model.generate_symbols(length_bars * BEATS_PER_BAR * 4, temperature, rng)
        melody = symbols_to_events(symbols, scale, getattr(model, "encoding", "c_octave_v1"))
        if cfg["harmonic_sevenths"] and str(variation).strip().lower() == "harmonic":
            melody = _harmonic_sevenths(melody, scale, bar_scales)
        if cfg["snap_melody"]:
            melody = _snap_to_harmony(melody, bar_scales, chord_pcs)
        if centred:
            melody = center_melody(melody, cfg["melody_center"] + 12 * octave)
    melody = theory.clip_events(melody, total_beats)

    pattern_library = pattern_library or {}
    ranges = cfg["pattern_ranges"]
    bass_pat = pat.choose(pattern_library, "bass", bass_choice, rng)
    arp_pat = pat.choose(pattern_library, "arpeggio", arp_choice, rng)
    harmony_patterns = cfg["harmony"] == "patterns"
    harm_pat = pat.choose(pattern_library, "harmony", harmony_choice, rng) if harmony_patterns else None

    def apply(p, part):
        if cfg["split_held"]:
            return pat.apply_pattern(p, part, roots, scale, BEATS_PER_BAR, ranges, bar_scales, split_held=True)
        return pat.apply_pattern(p, part, roots, scale, BEATS_PER_BAR, ranges, bar_scales)

    bass = apply(bass_pat, "bass") if bass_pat else theory.bass_events(roots, BEATS_PER_BAR)
    arpeggio = apply(arp_pat, "arpeggio") if arp_pat else theory.arpeggio_events(roots, scale, BEATS_PER_BAR)
    if harmony_patterns:
        harmony = apply(harm_pat, "harmony") if harm_pat else _block_chords(chord_pcs, BEATS_PER_BAR)
        if cfg["chords"] == "extended":  # original chords keep their names
            labels = _names_for_voicings(prog, roots, scale, variation, harmony, labels)
    else:
        harmony = theory.harmony_events(roots, scale, BEATS_PER_BAR)

    extra, extra_info = {}, {}
    if "arpeggio2" in cfg["parts"]:
        arp2_pat, shift = _pick_second_arpeggio(pattern_library.get("arpeggio", []), arp_pat, arp2_choice, rng)
        arp2 = [(o, p + shift, d) for o, p, d in apply(arp2_pat, "arpeggio")] if arp2_pat else []
        extra["arpeggio2"] = theory.clip_events(arp2, total_beats)
        extra_info["arp2_pattern"] = arp2_pat["id"] if arp2_pat else None
    if "melody2" in cfg["parts"]:
        counter_t = techno_melody.choose(templates, "slow", "random", rng)
        center = (cfg["melody_center"] or 72) + 12 * octave
        extra["melody2"] = theory.clip_events(
            techno_melody.make_counter(counter_t, melody, roots, chord_pcs, bar_scales, length_bars, rng, center),
            total_beats)
        extra_info["counter_template"] = counter_t["id"] if counter_t else "un acorde por nota"
    if lead_t is not None:
        extra_info["lead_template"] = lead_t["id"]

    loop_bars = min(length_bars, math.lcm(n_loop, 4)) if cfg["export_loops"] else length_bars
    chosen_bpm = float(bpm) if (cfg["bpm_range"] and bpm) else (cfg["fixed_bpm"] or BPM)
    chosen_energy = (float(energy) if (cfg["energy_range"] and energy is not None)
                     else (cfg["fixed_energy"] if cfg["fixed_energy"] is not None else ENERGY))

    generated = {
        "melody": melody,
        "arpeggio": theory.clip_events(arpeggio, total_beats),
        "bass": theory.clip_events(bass, total_beats),
        "harmony": theory.clip_events(harmony, total_beats),
    }
    generated.update(extra)
    missing = [p for p in cfg["parts"] if p not in generated]
    if missing:
        raise NotImplementedError(f"el estilo {style!r} pide partes que aún no se generan: {missing}")

    return {
        "style": style,
        "parts": tuple(cfg["parts"]),
        "loop_bars": loop_bars,
        "bpm": chosen_bpm,
        "energy": chosen_energy,
        "octave": octave if centred else 0,
        "bass_choice": bass_choice, "arp_choice": arp_choice, "harmony_choice": harmony_choice,
        "bass_pattern": bass_pat["id"] if bass_pat else pat.BASIC,
        "arp_pattern": arp_pat["id"] if arp_pat else pat.BASIC,
        "harmony_pattern": (harm_pat["id"] if harm_pat else pat.BASIC) if harmony_patterns else None,
        "key": key, "mode": mode, "variation": variation, "mood": mood,
        **({"substyle": substyle} if substyle and cfg.get("substyles") else {}),
        "length_bars": length_bars, "temperature": temperature, "seed": seed,
        "scale": scale, "total_beats": total_beats,
        "chords": labels[:n_loop],  # one loop of the progression (sheet format)
        "chords_full": labels,
        "template_source": prog["source"], "template_degrees": prog["degrees"],
        "events": {p: generated[p] for p in cfg["parts"]},
        **extra_info,
    }


def _prefer_substyle(substyle, progressions, pattern_library, mood, mode, group_of):
    """Progressions and patterns of the songs of one sub-style, falling back to all of
    them wherever the sub-style has nothing (e.g. no slow melody in DnB)."""
    sub = str(substyle).strip().lower()
    key = group_of or (lambda m: m)
    mine = [p for p in progressions if p.get("style") == sub]
    if mine:
        same = [p for p in mine if key(p["mood"]) == key(mood.lower()) and p["mode"] == mode.lower()]
        same = same or [p for p in mine if p["mode"] == mode.lower()]
        progressions = same or mine
    lib = dict(pattern_library or {})
    owner = lib.get("substyles", {})
    if owner:
        for part, pool in list(lib.items()):
            if part == "substyles" or not isinstance(pool, list):
                continue
            if part == "melody_templates":
                kept = []
                for kind in ("hook", "slow"):
                    all_kind = [t for t in pool if t["kind"] == kind]
                    kept += [t for t in all_kind if owner.get(t["source"]) == sub] or all_kind
                lib[part] = kept
            else:
                lib[part] = [p for p in pool if owner.get(p["source"]) == sub] or pool
    return progressions, lib


def _register(pattern):
    pitches = [n[1] for b in pattern["bars"] for n in b["notes"]]
    return sum(pitches) / len(pitches)


def _pick_second_arpeggio(pool, first, choice, rng):
    """Second arpeggio with a different role from the first -> (pattern, octave shift).

    If the first one comes from a song that already pairs two arpeggios (002), use
    its partner, as the user wrote it. Otherwise prefer the pattern farthest in
    register and density; if both still sit in the same register, the second
    moves an octave away (up when it is the busier one)."""
    if not pool:
        return None, 0
    if choice not in (None, "auto", "random"):
        partner = next((p for p in pool if p["id"] == choice), None)
    else:
        partner = None
        if first is not None:
            partner = next((p for p in pool if p["source"] == first["source"] and p["id"] != first["id"]), None)
        if partner is None:
            others = [p for p in pool if first is None or p["source"] != first["source"]] or pool
            if first is None:
                partner = others[int(rng.integers(len(others)))]
            else:
                def distance(p):
                    d_reg = abs(_register(p) - _register(first)) / 12
                    d_den = abs(p["notes_per_bar"] - first["notes_per_bar"]) / max(p["notes_per_bar"], first["notes_per_bar"])
                    return d_reg + d_den
                scores = np.array([distance(p) for p in others]) + 1e-6
                partner = others[int(rng.choice(len(others), p=scores / scores.sum()))]
    shift = 0
    if first is not None and partner is not None and abs(_register(partner) - _register(first)) < 5:
        shift = 12 if partner["notes_per_bar"] >= first["notes_per_bar"] else -12
    return partner, shift


def song_parts(song):
    """Parts of a generated song, in the style's order."""
    return song.get("parts") or styles.get(song.get("style", AMBIENT))["parts"]


def is_melodic(part):
    return part.startswith("melody")


def file_prefix(song):
    return styles.get(song.get("style", AMBIENT))["file_prefix"]


def midi_files(song, tag):
    """{filename: bytes}: melody_021.mid … (ambient) or lofi_melody_021.mid … (lo-fi).

    Styles with export_loops export the accompaniment as its repeating loop
    (4 bars), like the hand-made dataset; the melody keeps the full length."""
    cfg = styles.get(song.get("style", AMBIENT))
    pre = cfg["file_prefix"]
    loop_beats = song.get("loop_bars", song["length_bars"]) * BEATS_PER_BAR
    files = {}
    for part in song_parts(song):
        events = song["events"][part]
        if not is_melodic(part) and cfg["export_loops"]:
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
    cfg = styles.get(song.get("style", AMBIENT))
    pre = cfg["file_prefix"]
    bpm = song.get("bpm", BPM)
    return {
        "Name": cfg["name_format"].format(n=number),
        "Key": song["key"].lower(),
        "mode": song["mode"],
        "variation": song["variation"],
        "bpm": int(bpm) if float(bpm).is_integer() else bpm,
        "TimSig": TIME_SIG,
        "mood": song["mood"],
        "energy": song.get("energy", ENERGY),
        "style": song.get("substyle") or cfg["sheet_style"],
        "chords": "-".join(song["chords"]),
        # one column per part, named like the part (melody2, arpeggio2… in techno)
        **{part: f"{pre}{part}_{tag}.mid" for part in song_parts(song)},
    }
