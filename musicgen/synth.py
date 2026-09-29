"""Tiny NumPy synthesizer so the song can be heard in the browser without
FluidSynth or soundfonts. Each part gets a simple timbre; the parts are
rendered separately and mixed with per-track volumes into a 16-bit WAV."""

import io
import wave

import numpy as np

SR = 22050

# name -> (harmonic amplitudes, attack s, decay time-constant s (None = sustain), release s)
TIMBRES = {
    "melody":   ([1.0, 0.35, 0.12, 0.05], 0.012, 1.6, 0.18),
    "arpeggio": ([1.0, 0.0, 0.30, 0.0, 0.12], 0.004, 0.35, 0.08),
    "bass":     ([1.0, 0.25, 0.05], 0.03, None, 0.12),
    "harmony":  ([1.0, 0.2, 0.08], 0.25, None, 0.4),
}


def midi_to_hz(p):
    return 440.0 * 2 ** ((p - 69) / 12)


def render_part(events, bpm, part, total_beats):
    harmonics, attack, decay, release = TIMBRES[part]
    sec_per_beat = 60.0 / bpm
    n_total = int((total_beats * sec_per_beat + 1.0) * SR)
    out = np.zeros(n_total, dtype=np.float32)
    for start, pitch, dur in events:
        s0 = int(start * sec_per_beat * SR)
        n_on = max(1, int(dur * sec_per_beat * SR))
        n_rel = int(release * SR)
        n = min(n_on + n_rel, n_total - s0)
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / SR
        f = midi_to_hz(pitch)
        wave_ = np.zeros(n, dtype=np.float32)
        for k, a in enumerate(harmonics, start=1):
            if a and f * k < SR / 2:
                wave_ += a * np.sin(2 * np.pi * f * k * t)
        env = np.minimum(1.0, t / attack)
        if decay:
            env *= np.exp(-t / decay) * 0.8 + 0.2
        rel_t = np.clip((t - n_on / SR) / release, 0, 1)
        env *= 1.0 - rel_t
        out[s0:s0 + n] += wave_ * env
    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def mix_to_wav(tracks, volumes):
    """tracks: {part: np.array}, volumes: {part: 0..1} -> WAV bytes."""
    n = max(len(a) for a in tracks.values())
    mix = np.zeros(n, dtype=np.float32)
    for part, audio in tracks.items():
        v = volumes.get(part, 0.0)
        if v > 0:
            mix[:len(audio)] += v * audio
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.9
    pcm = (mix * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
