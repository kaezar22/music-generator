"""MIDI read/write in memory (bytes), so nothing touches the server's disk."""

import io

import mido
from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from .theory import STEP

TPB = 96  # same resolution as the existing song files


def events_to_midi_bytes(events, bpm, velocity=100, program=None):
    mid = MidiFile(ticks_per_beat=TPB)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))
    track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    if program is not None:
        track.append(Message("program_change", program=program, time=0))
    abs_ev = []
    for start, pitch, dur in events:
        on = round(start * TPB)
        off = max(on + 1, round((start + dur) * TPB))
        abs_ev.append((on, 1, pitch))
        abs_ev.append((off, 0, pitch))
    abs_ev.sort(key=lambda e: (e[0], e[1]))  # note-off before note-on on ties
    last = 0
    for tick, on, pitch in abs_ev:
        kind = "note_on" if on else "note_off"
        track.append(Message(kind, note=pitch, velocity=velocity if on else 64, time=tick - last))
        last = tick
    track.append(MetaMessage("end_of_track", time=0))
    buf = io.BytesIO()
    mid.save(file=buf)
    return buf.getvalue()


def read_notes(source, top_voice_only=True):
    """[(onset_beats, pitch, dur_beats)] from a path, bytes or file-like.

    With top_voice_only the highest note of each 16th-grid onset is kept
    (melody view)."""
    if isinstance(source, (bytes, bytearray)):
        mid = MidiFile(file=io.BytesIO(source))
    elif hasattr(source, "read"):
        mid = MidiFile(file=source)
    else:
        mid = MidiFile(source)
    tpb = mid.ticks_per_beat
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type in ("note_on", "note_off"):
                on = msg.type == "note_on" and msg.velocity > 0
                events.append((t, 1 if on else 0, msg.note))
    events.sort()
    open_notes, notes = {}, []
    for t, on, p in events:
        if on:
            open_notes.setdefault(p, []).append(t)
        elif open_notes.get(p):
            s = open_notes[p].pop(0)
            notes.append((s / tpb, p, (t - s) / tpb))
    notes.sort()
    if not top_voice_only:
        return notes
    by_onset = {}
    for on, p, d in notes:
        q = round(on / STEP)
        if q not in by_onset or p > by_onset[q][1]:
            by_onset[q] = (q * STEP, p, d)
    return [by_onset[q] for q in sorted(by_onset)]
