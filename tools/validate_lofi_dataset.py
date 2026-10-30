# -*- coding: utf-8 -*-
"""Valida el dataset lo-fi ANTES de entrenar.

Lee las filas del Google Sheet (pestaña lofi) y los MIDI de lofi_songs/song_XXX,
y reporta todo lo que el parser no reconoce o que rompe las convenciones:

  ERROR  -> hay que corregirlo antes de entrenar
  AVISO  -> revisar; puede ser intencional
  INFO   -> datos útiles para decidir el preprocesamiento

Usa el mismo parser que el entrenamiento y la app (musicgen/theory.py).

Uso (desde la carpeta musicai_webapp):
    python tools/validate_lofi_dataset.py
    python tools/validate_lofi_dataset.py --songs ..\\lofi_songs --credentials ..\\AI_MUSIC_Project\\credentials.json
    python tools/validate_lofi_dataset.py --csv lofi.csv        # sin internet: CSV exportado de la pestaña

Sale con código 1 si hay errores.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import Counter, defaultdict

import mido

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from musicgen import theory  # noqa: E402

SPREADSHEET_ID = "1z_njzCTDkI59D6dggXbFxONousXaHWe1wGoilxFo-Pc"
LOFI_GID = 913708363
EXPECTED_COLUMNS = ["Name", "Key", "mode", "variation", "bpm", "TimSig", "mood", "energy",
                    "style", "chords", "melody", "harmony", "arpeggio", "bass"]
PARTS = ("melody", "harmony", "arpeggio", "bass")
LOOP_BARS, MELODY_BARS, BEATS_PER_BAR = 4, 16, 4
BPM_RANGE, ENERGY_RANGE = (70, 90), (0.25, 0.6)
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def nn(p):
    return f"{NOTE[p % 12]}{p // 12 - 1}"


class Report:
    def __init__(self):
        self.items = defaultdict(list)  # song -> [(level, msg)]
        self.global_items = []

    def add(self, song, level, msg):
        (self.items[song] if song else self.global_items).append((level, msg))

    def count(self, level):
        n = sum(1 for lv, _ in self.global_items if lv == level)
        return n + sum(1 for v in self.items.values() for lv, _ in v if lv == level)


# ------------------------------------------------------------------ Sheet
def read_sheet_rows(args, rep):
    """-> list of (row_number, {col: raw_value}) with the RAW cell text."""
    if args.csv:
        with open(args.csv, newline="", encoding="utf-8-sig") as f:
            values = list(csv.reader(f))
        source = args.csv
    else:
        import gspread
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_file(
            args.credentials, scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
        ws = gspread.authorize(creds).open_by_key(SPREADSHEET_ID).get_worksheet_by_id(args.gid)
        values = ws.get_all_values()
        source = f"Google Sheet, pestaña '{ws.title}' (gid {args.gid})"
    rep.add(None, "INFO", f"Metadata leída de: {source}")
    if not values:
        rep.add(None, "ERROR", "La pestaña está vacía.")
        return []
    header_raw = values[0]
    header = [h.strip() for h in header_raw]
    for h_raw, h in zip(header_raw, header):
        if h_raw != h:
            rep.add(None, "AVISO", f"Encabezado con espacios: {h_raw!r}")
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    extra = [c for c in header if c and c not in EXPECTED_COLUMNS]
    if missing:
        rep.add(None, "ERROR", f"Faltan columnas: {missing}")
    if extra:
        rep.add(None, "AVISO", f"Columnas no esperadas (se ignoran): {extra}")
    rows = []
    for i, raw in enumerate(values[1:], start=2):
        row = {h: (raw[j] if j < len(raw) else "") for j, h in enumerate(header) if h}
        if not any(str(v).strip() for v in row.values()):
            continue
        rows.append((i, row))
    return rows


def song_number(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def check_row(rownum, raw, rep):
    """Validates one Sheet row. Returns the parsed dict (or None if unusable)."""
    name = str(raw.get("Name", "")).strip()
    num = song_number(name)
    song = f"song_{num:03d}" if num is not None else f"fila {rownum}"
    row = {}
    for col in EXPECTED_COLUMNS:
        v = raw.get(col, "")
        sv = str(v).strip()
        if v != sv:
            rep.add(song, "INFO", f"Celda {col!r} con espacios al inicio/final ({v!r}); se aplica strip().")
        row[col] = sv
        if sv == "":
            rep.add(song, "ERROR", f"Celda vacía: {col}")
    if num is None:
        rep.add(song, "ERROR", f"Name sin número reconocible: {name!r}")
        return None
    parsed = {"num": num, "song": song, "row": row, "rownum": rownum}
    try:
        parsed["scale"] = theory.parse_scale(row["Key"], row["mode"], row["variation"])
    except (ValueError, KeyError) as e:
        rep.add(song, "ERROR", str(e))
    try:
        bpm = float(row["bpm"].replace(",", "."))
        parsed["bpm"] = bpm
        if not BPM_RANGE[0] <= bpm <= BPM_RANGE[1]:
            rep.add(song, "AVISO", f"bpm {bpm:g} fuera del rango {BPM_RANGE}")
    except ValueError:
        rep.add(song, "ERROR", f"bpm no numérico: {row['bpm']!r}")
    if row["TimSig"] != "4/4":
        rep.add(song, "ERROR", f"TimSig {row['TimSig']!r}; se espera 4/4")
    try:
        energy = float(row["energy"].replace(",", "."))
        if not ENERGY_RANGE[0] <= energy <= ENERGY_RANGE[1]:
            rep.add(song, "AVISO", f"energy {energy:g} fuera del rango {ENERGY_RANGE}")
    except ValueError:
        rep.add(song, "ERROR", f"energy no numérica: {row['energy']!r}")
    if row["style"].lower() != "lofi":
        rep.add(song, "AVISO", f"style {row['style']!r}; se espera 'lofi'")
    try:
        chords = theory.parse_chords(row["chords"])
        parsed["chords"] = chords
        if len(chords) != LOOP_BARS:
            rep.add(song, "ERROR", f"{len(chords)} acordes en 'chords'; se esperan {LOOP_BARS} (uno por compás)")
        if "scale" in parsed:
            off = [c["label"] for c in chords if c["root_pc"] not in parsed["scale"]]
            if off:
                rep.add(song, "AVISO", f"Acordes con fundamental fuera de la escala: {off}")
    except ValueError as e:
        rep.add(song, "ERROR", str(e))
    for part in PARTS:
        expected = f"lofi_{part}_{num:03d}.mid"
        if row[part] and row[part] != expected:
            rep.add(song, "ERROR", f"Columna {part}: {row[part]!r}; se espera {expected!r}")
    return parsed


# ------------------------------------------------------------------ MIDI
def read_midi(path):
    mid = mido.MidiFile(path)
    info = {"type": mid.type, "tpb": mid.ticks_per_beat, "tempos": [], "timesigs": [],
            "track_names": [], "velocities": []}
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                info["tempos"].append((t, mido.tempo2bpm(msg.tempo)))
            elif msg.type == "time_signature":
                info["timesigs"].append((t, f"{msg.numerator}/{msg.denominator}"))
            elif msg.type == "track_name" and msg.name.strip():
                info["track_names"].append(msg.name)
            elif msg.type in ("note_on", "note_off"):
                on = msg.type == "note_on" and msg.velocity > 0
                events.append((t, 1 if on else 0, msg.note, msg.velocity))
    events.sort(key=lambda e: (e[0], e[1]))
    open_notes, notes = {}, []
    for t, on, p, v in events:
        if on:
            open_notes.setdefault(p, []).append((t, v))
            info["velocities"].append(v)
        elif open_notes.get(p):
            s, v0 = open_notes[p].pop(0)
            notes.append((s, p, t - s, v0))
    info["notes"] = sorted(notes)  # (start_tick, pitch, dur_ticks, vel)
    return info


def allowed_pcs(scale, chord, variation, key_pc):
    """Notas permitidas en un compás: escala + notas del acorde.

    En menor armónica se aceptan las DOS versiones de la 7ª (natural y elevada);
    dónde aparece cada una se revisa aparte (harmonic_seventh_notice)."""
    allowed = set(scale) | chord["pcs"]
    if variation == "harmonic":
        allowed |= {(key_pc + 10) % 12, (key_pc + 11) % 12}
    return allowed


def harmonic_seventh_notice(pitch, chord, key_pc):
    """Menor armónica: la 7ª elevada se espera sobre el V7 y la natural en el resto.
    Devuelve un texto si la nota se sale de esa costumbre (para un AVISO)."""
    raised7, natural7 = (key_pc + 11) % 12, (key_pc + 10) % 12
    on_dominant = raised7 in chord["pcs"]
    pc = pitch % 12
    if pc == raised7 and not on_dominant:
        return "7ª elevada (sensible) fuera del V7"
    if pc == natural7 and on_dominant:
        return "7ª natural sobre el V7 (suena como #9 del dominante)"
    return None


def check_midi(parsed, folder, rep, stats):
    song, row = parsed["song"], parsed["row"]
    num = parsed["num"]
    scale, chords = parsed.get("scale"), parsed.get("chords")
    variation = row["variation"].lower()
    key_pc = theory.parse_key(row["Key"]) if "scale" in parsed else None
    expected_files = {f"lofi_{p}_{num:03d}.mid" for p in PARTS}
    present = {os.path.basename(f) for f in glob.glob(os.path.join(folder, "*"))}
    for extra in sorted(present - expected_files):
        rep.add(song, "AVISO", f"Archivo sobrante en la carpeta: {extra}")
    for p in PARTS:
        fname = f"lofi_{p}_{num:03d}.mid"
        path = os.path.join(folder, fname)
        if not os.path.isfile(path):
            rep.add(song, "ERROR", f"Falta el archivo {fname}")
            continue
        try:
            info = read_midi(path)
        except Exception as e:  # noqa: BLE001
            rep.add(song, "ERROR", f"{fname}: no se pudo leer ({e})")
            continue
        tpb = info["tpb"]
        tag = p
        if info["type"] != 1:
            rep.add(song, "AVISO", f"{tag}: MIDI type {info['type']} (se esperaba 1)")
        if tpb != 96:
            rep.add(song, "AVISO", f"{tag}: ticks_per_beat {tpb} (se esperaba 96)")
        if not info["tempos"]:
            rep.add(song, "ERROR", f"{tag}: sin tempo en el MIDI")
        else:
            bpms = sorted({round(b, 2) for _, b in info["tempos"]})
            if len(bpms) > 1:
                rep.add(song, "AVISO", f"{tag}: varios tempos {bpms}")
            if "bpm" in parsed and abs(bpms[0] - parsed["bpm"]) > 0.5:
                rep.add(song, "ERROR", f"{tag}: tempo {bpms[0]:g} en el MIDI ≠ bpm {parsed['bpm']:g} del Sheet")
        ts = {s for _, s in info["timesigs"]}
        if ts and ts != {"4/4"}:
            rep.add(song, "ERROR", f"{tag}: compás {sorted(ts)} (se espera 4/4)")
        stats["track_names"][p].update(info["track_names"])
        if info["velocities"]:
            stats["velocities"].update(info["velocities"])
        notes = info["notes"]
        if not notes:
            rep.add(song, "ERROR", f"{tag}: sin notas")
            continue
        bar_ticks = BEATS_PER_BAR * tpb
        end = max(s + d for s, _, d, _ in notes)
        bars = math.ceil(end / bar_ticks - 1e-9)
        want = MELODY_BARS if p == "melody" else LOOP_BARS
        if bars != want:
            rep.add(song, "ERROR", f"{tag}: dura {end / bar_ticks:.2f} compases; se esperan {want}")
        sixteenth = tpb // 4
        offgrid = [(s, pi) for s, pi, d, _ in notes if s % sixteenth or d % sixteenth]
        if offgrid:
            lvl = "ERROR" if p == "melody" else "AVISO"
            ex = ", ".join(f"{nn(pi)} en tiempo {s / tpb + 1:.3f}" for s, pi in offgrid[:3])
            rep.add(song, lvl, f"{tag}: {len(offgrid)} notas fuera de la cuadrícula de semicorcheas ({ex})")
        if p == "melody":
            overlaps = [(a, b) for a, b in zip(notes, notes[1:]) if a[0] + a[2] > b[0]]
            same_onset = Counter(s for s, *_ in notes)
            if overlaps or any(c > 1 for c in same_onset.values()):
                rep.add(song, "ERROR", f"melodía no monofónica ({len(overlaps)} solapamientos)")
            lo, hi = min(pi for _, pi, _, _ in notes), max(pi for _, pi, _, _ in notes)
            stats["melody_range"][song] = (lo, hi, len(notes))
        if not (scale and chords and len(chords) == LOOP_BARS):
            continue
        # notas fuera de escala/acorde, compás por compás (loop alineado con la melodía)
        bad, tensions, sevenths = [], 0, []
        by_bar = defaultdict(list)
        for s, pi, d, v in notes:
            bar = int(s // bar_ticks)
            ch = chords[bar % LOOP_BARS]
            where = f"compás {bar + 1} tiempo {(s % bar_ticks) / tpb + 1:g}"
            if pi % 12 not in allowed_pcs(scale, ch, variation, key_pc):
                bad.append(f"{nn(pi)} en {where} ({ch['label']})")
            else:
                if variation == "harmonic":
                    why = harmonic_seventh_notice(pi, ch, key_pc)
                    if why:
                        sevenths.append(f"{nn(pi)} en {where} sobre {ch['label']}: {why}")
                if pi % 12 not in ch["pcs"]:
                    tensions += 1
            by_bar[bar].append((s, pi, d))
        for msg in sevenths:
            rep.add(song, "AVISO", f"{tag}: {msg}")
        if bad:
            rep.add(song, "ERROR", f"{tag}: {len(bad)} notas fuera de escala/acorde: " + "; ".join(bad[:4])
                    + (" …" if len(bad) > 4 else ""))
        if p == "harmony":
            attacks = [len({s for s, _, _ in by_bar[b]}) for b in range(LOOP_BARS)]
            if max(attacks, default=0) <= 1:
                rep.add(song, "AVISO", "armonía sin comping (un solo golpe por compás)")
            for b in range(LOOP_BARS):
                ch = chords[b]
                pcs = {pi % 12 for _, pi, _ in by_bar[b]}
                extra = pcs - ch["pcs"]
                if len(extra) > 1:
                    rep.add(song, "ERROR", f"armonía compás {b + 1}: no coincide con {ch['label']} "
                                           f"(notas {sorted(NOTE[x] for x in extra)} fuera del acorde)")
                elif extra:
                    rep.add(song, "AVISO", f"armonía compás {b + 1}: la nota {NOTE[extra.pop()]} "
                                           f"no está en {ch['label']} (¿tensión no escrita en el nombre?)")
                if ch["third_pc"] is not None and pcs and ch["third_pc"] not in pcs:
                    rep.add(song, "AVISO", f"armonía compás {b + 1}: el voicing no tiene la 3ª de {ch['label']}")
        if p == "bass":
            for b in range(LOOP_BARS):
                if not by_bar[b]:
                    rep.add(song, "ERROR", f"bajo: compás {b + 1} vacío")
                    continue
                first = min(by_bar[b])
                want = chords[b]["bass_pc"]
                if all(pi % 12 != want for _, pi, _ in by_bar[b]):
                    rep.add(song, "ERROR", f"bajo compás {b + 1}: nunca toca {NOTE[want]}, "
                                           f"la nota del bajo de {chords[b]['label']}")
                elif first[1] % 12 != want:
                    rep.add(song, "AVISO", f"bajo compás {b + 1}: empieza en {NOTE[first[1] % 12]}, "
                                           f"no en {NOTE[want]} ({chords[b]['label']})")
        if p in ("melody", "arpeggio") and tensions:
            stats["tensions"][song][p] = tensions


def run_validation(args):
    """Lee Sheet + MIDI y valida. -> (by_song, folders, rep, stats).

    by_song: {"song_001": {"num", "song", "row", "scale", "chords", "bpm", ...}}"""
    rep = Report()
    stats = {"track_names": defaultdict(Counter), "velocities": Counter(),
             "melody_range": {}, "tensions": defaultdict(dict)}
    rows = read_sheet_rows(args, rep)
    parsed_rows = [p for p in (check_row(n, r, rep) for n, r in rows) if p]
    by_song = {}
    for p in parsed_rows:
        if p["song"] in by_song:
            rep.add(p["song"], "ERROR", f"Name repetido en las filas {by_song[p['song']]['rownum']} y {p['rownum']}")
        by_song[p["song"]] = p
    folders = {os.path.basename(f): f for f in glob.glob(os.path.join(args.songs, "song_*")) if os.path.isdir(f)}
    for name in sorted(set(folders) - set(by_song)):
        rep.add(name, "ERROR", "Carpeta sin fila en el Sheet")
    for name in sorted(set(by_song) - set(folders)):
        rep.add(name, "ERROR", f"Fila del Sheet sin carpeta {name} en {args.songs}")
    for name in sorted(set(by_song) & set(folders)):
        check_midi(by_song[name], folders[name], rep, stats)
    return by_song, folders, rep, stats


def add_source_args(ap):
    ap.add_argument("--songs", default=os.path.join(REPO, "..", "lofi_songs"),
                    help="carpeta con song_001 … song_020 (por defecto ../lofi_songs)")
    ap.add_argument("--credentials", default=os.path.join(REPO, "..", "AI_MUSIC_Project", "credentials.json"))
    ap.add_argument("--gid", type=int, default=LOFI_GID, help="pestaña del Sheet (por defecto la lo-fi)")
    ap.add_argument("--csv", help="usar un CSV exportado de la pestaña en lugar del Google Sheet")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    args = ap.parse_args()
    by_song, folders, rep, stats = run_validation(args)
    parsed_rows = list(by_song.values())

    # ---------- informe
    print("=" * 78)
    print(f"VALIDACIÓN DATASET LO-FI · {len(by_song)} filas en el Sheet · {len(folders)} carpetas")
    print("=" * 78)
    for lv, msg in rep.global_items:
        print(f"[{lv}] {msg}")
    for song in sorted(rep.items):
        print(f"\n{song}")
        for lv, msg in sorted(rep.items[song], key=lambda x: ["ERROR", "AVISO", "INFO"].index(x[0])):
            print(f"  [{lv}] {msg}")

    print("\n" + "-" * 78 + "\nRESUMEN PARA EL PREPROCESAMIENTO\n" + "-" * 78)
    moods = Counter(p["row"]["mood"].lower() for p in parsed_rows)
    print(f"Moods ({len(moods)} distintos): " + ", ".join(f"{m}×{c}" for m, c in moods.most_common()))
    var = Counter(f"{p['row']['mode']}/{p['row']['variation']}" for p in parsed_rows)
    print("Modo/variación: " + ", ".join(f"{k}×{c}" for k, c in sorted(var.items())))
    qual = Counter(c["quality"] or "(mayor)" for p in parsed_rows for c in p.get("chords", []))
    print("Tipos de acorde: " + ", ".join(f"{k}×{c}" for k, c in qual.most_common()))
    if stats["velocities"]:
        v = stats["velocities"]
        total = sum(v.values())
        print(f"Velocities: de {min(v)} a {max(v)} · {total} notas, {total - v.get(100, 0)} con velocity ≠ 100 "
              f"· valores usados: {sorted(v)}")
    for p in PARTS:
        if stats["track_names"][p]:
            print(f"Nombres de pista ({p}): " + ", ".join(f"{n!r}×{c}" for n, c in stats["track_names"][p].items()))
    if stats["melody_range"]:
        print("\nRegistro de la melodía:")
        for song, (lo, hi, n) in sorted(stats["melody_range"].items()):
            center = (lo + hi) / 2
            print(f"  {song}: {nn(lo):>4} – {nn(hi):<4} ({hi - lo:2d} semitonos, {n:3d} notas, centro {nn(round(center))})")
    if stats["tensions"]:
        print("\nNotas de color (en la escala pero fuera del acorde; NO son errores):")
        for song, d in sorted(stats["tensions"].items()):
            print(f"  {song}: " + ", ".join(f"{k} {v}" for k, v in d.items()))

    ne, na = rep.count("ERROR"), rep.count("AVISO")
    print("\n" + "=" * 78)
    print(f"RESULTADO: {ne} errores · {na} avisos")
    print("Listo para entrenar." if ne == 0 else "Corrige los errores antes de entrenar.")
    print("=" * 78)
    return 1 if ne else 0


if __name__ == "__main__":
    sys.exit(main())
