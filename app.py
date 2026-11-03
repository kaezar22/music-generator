"""MusicAI — generador de canciones (melodía LSTM + arpegio + bajo + armonía).

Flujo: elegir estilo y escala / mood / longitud / temperatura -> generar ->
verificar originalidad de la melodía -> escuchar -> guardar (fila en Google
Sheet + zip). Dos estilos: ambient (la app original) y lo-fi (dataset propio).
"""

import csv
import os
import re
import secrets as pysecrets
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from musicgen import originality, patterns, pipeline, sheets, styles, synth, techno_melody, theory
from musicgen.melody import MelodyModel
from musicgen.moods import GROUP_LABELS, grouper

APP_DIR = os.path.dirname(os.path.abspath(__file__))
SPREADSHEET_ID = "1z_njzCTDkI59D6dggXbFxONousXaHWe1wGoilxFo-Pc"

STYLES = styles.STYLES
OCTAVES = {-1: "Más grave", 0: "Normal (≈ C5)", 1: "Más aguda"}

LENGTHS = [8, 16, 32, 64]
PART_LABELS = styles.PART_LABELS
PART_COLORS = styles.PART_COLORS
VERDICT_ICON = {"ORIGINAL": "✅", "SIMILAR": "⚠️", "COPIA": "⛔"}


def now_str():
    return datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M")


st.set_page_config(page_title="MusicAI", page_icon="🎵", layout="wide")


# ----------------------------------------------------------------- password
def require_password():
    if st.session_state.get("auth_ok"):
        return
    expected = str(st.secrets.get("app_password", ""))
    st.title("🎵 MusicAI")
    with st.form("login"):
        pw = st.text_input("Contraseña", type="password")
        ok = st.form_submit_button("Entrar", type="primary")
    if ok:
        if expected and pw == expected:
            st.session_state.auth_ok = True
            st.rerun()
        elif not expected:
            st.error("La app no tiene contraseña configurada (falta `app_password` en los secrets).")
        else:
            st.error("Contraseña incorrecta.")
    st.stop()


require_password()


# ----------------------------------------------------------------- resources (por estilo)
def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [r for r in rows if r.get("Name")]


@st.cache_resource
def get_model(style):
    """LSTM of the style, or None for styles whose melodies come from templates (techno)."""
    models = STYLES[style]["models"]
    if not models or not os.path.isfile(os.path.join(models, "melody_lstm_weights.npz")):
        return None
    return MelodyModel(models)


@st.cache_resource
def get_worksheet(style):
    if "gcp_service_account" not in st.secrets:
        return None
    cfg = STYLES[style]
    return sheets.open_worksheet(st.secrets["gcp_service_account"],
                                 st.secrets.get("spreadsheet_id", SPREADSHEET_ID),
                                 st.secrets.get(cfg["gid_secret"], cfg["gid_default"]))


@st.cache_data(ttl=300, show_spinner=False)
def load_metadata(style):
    """(rows, source). Sheet first; bundled CSV as fallback."""
    try:
        ws = get_worksheet(style)
        if ws is not None:
            return sheets.read_metadata(ws), "Google Sheet"
    except Exception as e:  # noqa: BLE001
        st.session_state[f"sheet_error_{style}"] = str(e)
    return read_csv_rows(os.path.join(STYLES[style]["library"], "metadata.csv")), "CSV incluido en la app"


@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_melodies(style):
    """Melodies saved by the app in the style's melodies tab: (rows, error)."""
    try:
        ws = get_worksheet(style)
        if ws is None:
            return [], None
        return sheets.read_melodies(ws, tab=STYLES[style]["melody_tab"]), None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


@st.cache_resource
def get_pattern_library(style):
    """Patterns extracted from the songs bundled with the app for this style."""
    cfg = STYLES[style]
    rows = read_csv_rows(os.path.join(cfg["library"], "metadata.csv"))
    lib = patterns.load_patterns(cfg["library"], rows, parts=cfg["pattern_parts"], **cfg["pattern_options"])
    if cfg.get("melody_source") in ("templates", "hybrid"):
        lib["melody_templates"] = techno_melody.load_templates(
            cfg["library"], rows, cfg["melody_columns"], cfg["pattern_options"].get("read_options"))
    if cfg.get("substyles"):
        lib["substyles"] = {f"song_{n:03d}": str(r.get("style", "")).strip().lower()
                            for n, r in metadata_by_num(rows).items()}
    return lib


def metadata_by_num(rows):
    out = {}
    for r in rows:
        n = sheets.song_number(r.get("Name"))
        if n is not None:
            out[n] = r
    return out


@st.cache_data(show_spinner=False)
def bundled_library(style, meta_rows):
    cfg = STYLES[style]
    if cfg.get("melody_columns"):  # techno: every melody column (melody, melody2, melody3)
        rows = read_csv_rows(os.path.join(cfg["library"], "metadata.csv"))
        return originality.load_column_library(cfg["library"], rows, cfg["melody_columns"], metadata_by_num(meta_rows),
                                               cfg["pattern_options"].get("read_options"))
    return originality.load_folder_library(cfg["library"], metadata_by_num(meta_rows))


def song_num_of(key):
    """'song_021' or 'song_021/melody2' -> 21."""
    m = re.match(r"song_(\d+)", key)
    return int(m.group(1)) if m else None


@st.cache_data(show_spinner=False)
def zip_library(zip_bytes, meta_rows):
    return originality.load_zip_library(zip_bytes, metadata_by_num(meta_rows))


def style_library(style, meta_rows):
    """{song: entry} bundled + guardadas en el Sheet + guardadas en esta sesión."""
    lib = dict(bundled_library(style, meta_rows))
    n_bundled = len(lib)
    rows, error = load_sheet_melodies(style)
    in_tab = set()
    for r in rows:
        in_tab.add(r["song"])
        if r["song"] in lib:
            continue
        try:
            scale = theory.scale_notes_ordered(r["key"], r["mode"], r.get("variation", "natural"))
        except Exception:  # noqa: BLE001
            scale = None
        lib[r["song"]] = originality.make_entry(r["song"], r["notes"], scale)
    lib.update(st.session_state.saved_songs.get(style, {}))
    return lib, n_bundled, in_tab, error


# ----------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("saved_songs", {})   # style -> {song: entry} saved in this session
ss.setdefault("song", None)
ss.setdefault("report", None)
ss.setdefault("saved", None)       # {"tag","zip","row"} for the current song


def on_style_change():
    for k in ("song", "report", "saved"):
        ss[k] = None
    ss.pop("tracks", None)


st.title("🎵 MusicAI — generador de canciones")
style = st.radio("Estilo", [k for k, v in STYLES.items() if v["enabled"]], horizontal=True, key="style",
                 on_change=on_style_change,
                 format_func=lambda s: STYLES[s]["label"])
cfg = STYLES[style]

meta_rows, meta_source = load_metadata(style)
progression_library = theory.build_progression_library(meta_rows)
model = get_model(style)
pattern_library = get_pattern_library(style)

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header(f"Biblioteca {cfg['label']}")
    library, n_bundled, in_sheet_tab, mel_error = style_library(style, meta_rows)
    st.caption(f"{n_bundled} incluidas en la app · {len(library) - n_bundled} guardadas en el Sheet "
               f"(pestaña «{cfg['melody_tab']}»).")
    if mel_error:
        st.error(f"No se pudo leer la pestaña de melodías: {mel_error}")
    up = st.file_uploader(
        "Agregar canciones (.zip)", type=["zip"], key=f"zip_{style}",
        help=f"Comprime tus carpetas song_XXX (o la carpeta {cfg['folder_hint']} completa) y súbela. "
             "Se usa para comparar melodías y para no repetir nombres de registro.",
    )
    if up is not None:
        extra = zip_library(up.getvalue(), meta_rows)
        new = [k for k in extra if k not in library]
        library.update(extra)
        st.caption(f"Zip: {len(extra)} melodías leídas ({len(new)} nuevas).")
        to_store = sorted(k for k in new if k not in in_sheet_tab)
        if to_store and get_worksheet(style) is not None:
            if st.button(f"Guardar {len(to_store)} melodía(s) en el Sheet", type="primary",
                         help="Así quedan en la biblioteca de forma permanente: " + ", ".join(to_store)):
                meta_num = metadata_by_num(meta_rows)
                for k in to_store:
                    num = song_num_of(k)
                    row = meta_num.get(num, {})
                    sheets.save_melody(get_worksheet(style), f"{num:03d}", row.get("Key", ""), row.get("mode", ""),
                                       row.get("variation", ""), extra[k]["notes"], now_str(),
                                       tab=cfg["melody_tab"])
                load_sheet_melodies.clear()
                st.rerun()
    own_library = dict(library)
    breakdown = [f"{len(own_library)} {cfg['label'].lower()}"]
    for other in cfg["compare_with"]:  # p. ej. las melodías lo-fi también se comparan contra las ambient
        other_meta, _ = load_metadata(other)
        other_lib, _, _, _ = style_library(other, other_meta)
        library.update({f"{other} {k}": v for k, v in other_lib.items()})
        breakdown.append(f"{len(other_lib)} {STYLES[other]['label'].lower()}")
    st.metric("Melodías para comparar", len(library),
              help=" + ".join(breakdown) if cfg["compare_with"] else None)
    registered = {f"song_{n:03d}" for n in metadata_by_num(meta_rows)}
    missing = sorted(registered - set(own_library))
    if missing:
        st.warning("Registradas en el Sheet pero sin melodía para comparar: " + ", ".join(missing)
                   + ". Sube sus zips arriba y presiona «Guardar … en el Sheet».")

    st.divider()
    st.header("Google Sheet")
    ws = None
    try:
        ws = get_worksheet(style)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo abrir el Sheet: {e}")
    if ws is None:
        st.warning("Sin conexión al Sheet: configura `gcp_service_account` en los secrets. "
                   "Se usa el CSV incluido como metadata.")
    else:
        st.caption(f"Conectado · {len(meta_rows)} registros {cfg['label'].lower()} leídos de: {meta_source}")
    if st.button("Recargar metadata"):
        load_metadata.clear()
        load_sheet_melodies.clear()
        st.rerun()

    st.divider()
    if st.button("Cerrar sesión"):
        ss.clear()
        st.rerun()


# ----------------------------------------------------------------- helpers
def used_numbers():
    nums = {sheets.song_number(r.get("Name")) for r in meta_rows}
    nums |= {song_num_of(k) for k in own_library}
    return {n for n in nums if n is not None}


def suggested_tag():
    return f"song_{max(used_numbers(), default=0) + 1:03d}"


def validate_tag(tag):
    m = re.fullmatch(r"song_(\d{3,})", tag.strip())
    if not m:
        return None, "El nombre debe tener el formato song_021."
    num = int(m.group(1))
    if num in used_numbers():
        return None, f"{tag} ya existe en el Sheet o en la biblioteca {cfg['label'].lower()}."
    return num, None


def run_generation(params, retry_until_ok, max_tries=10):
    tries = max_tries if retry_until_ok else 1
    for attempt in range(1, tries + 1):
        seed = pysecrets.randbits(32)
        song = pipeline.generate_song(model, progression_library, seed=seed,
                                      pattern_library=pattern_library, style=style, **params)
        report = originality.check_song(song, library, styles.melody_parts(cfg))
        if report["verdict"] != "COPIA":
            break
    song["attempts"] = attempt
    ss.song, ss.report, ss.saved = song, report, None
    ss.pop("tracks", None)


def pattern_name(pid):
    return "básico" if pid == patterns.BASIC else f"estilo {pid}"


def display_parts(song):
    """Parts of the song in display order (legend, sliders): melodies first, harmony last."""
    parts = pipeline.song_parts(song)
    return [p for p in styles.PART_LABELS if p in parts]


def piano_roll(song):
    rows = []
    shown = display_parts(song)
    for part in reversed(shown):  # harmony drawn first, melody on top
        for start, pitch, dur in song["events"][part]:
            rows.append({
                "Parte": PART_LABELS[part],
                "inicio": start / pipeline.BEATS_PER_BAR + 1,
                "fin": (start + dur) / pipeline.BEATS_PER_BAR + 1,
                "pitch": pitch, "pitch_top": pitch + 0.85,
                "Nota": f"{theory.NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}",
                "Compás": int(start // pipeline.BEATS_PER_BAR) + 1,
                "Duración (beats)": round(dur, 2),
            })
    df = pd.DataFrame(rows)
    order = [PART_LABELS[p] for p in shown]
    color = alt.Color("Parte:N", scale=alt.Scale(domain=order, range=[PART_COLORS[p] for p in shown]),
                      legend=alt.Legend(orient="top", title=None))
    chart = alt.Chart(df).mark_rect(cornerRadius=2).encode(
        x=alt.X("inicio:Q", title="Compás", scale=alt.Scale(domain=[1, song["length_bars"] + 1], nice=False),
                axis=alt.Axis(tickMinStep=1, grid=False)),
        x2="fin:Q",
        y=alt.Y("pitch:Q", title="Altura (MIDI)", scale=alt.Scale(zero=False)),
        y2="pitch_top:Q",
        color=color,
        opacity=alt.condition(alt.datum.Parte == "Armonía", alt.value(0.45), alt.value(0.95)),
        tooltip=["Parte", "Nota", "Compás", "Duración (beats)"],
    ).properties(height=340)
    return chart


def substyle_options(rows):
    """Sub-styles written in the Sheet's `style` column (techno: top, balad, DnB…),
    with the user's spelling, in order of appearance."""
    seen = {}
    for r in rows:
        v = str(r.get("style", "")).strip()
        if v and v.lower() not in seen and v.lower() != STYLES[style]["sheet_style"]:
            seen[v.lower()] = v
    return list(seen.values())


def substyle_tempo(rows, sub):
    """Median BPM / energy of the songs of a sub-style, or None."""
    vals = []
    for r in rows:
        if str(r.get("style", "")).strip().lower() == str(sub).lower():
            try:
                vals.append((float(r["bpm"]), float(r["energy"])))
            except (KeyError, TypeError, ValueError):
                pass
    if not vals:
        return None
    lo_b, hi_b = STYLES[style]["bpm_range"]
    lo_e, hi_e = STYLES[style]["energy_range"]
    bpm = int(min(hi_b, max(lo_b, np.median([b for b, _ in vals]))))
    energy = round(min(hi_e, max(lo_e, float(np.median([e for _, e in vals])))) * 20) / 20
    return bpm, energy


def suggest_bpm_energy(style_name, rows=None):
    """Al cambiar el mood o el subestilo, propone BPM y energy: el BPM del subestilo
    (DnB 140…) si lo hay, si no el del grupo de mood; la energy, según el grupo de mood."""
    cfg_ = STYLES[style_name]
    defaults = cfg_["group_defaults"]
    group = grouper(cfg_["mood_groups"])(ss.get(f"mood_{style_name}", ""))
    if group in defaults:
        ss[f"bpm_{style_name}"], ss[f"energy_{style_name}"] = defaults[group]
    if rows is not None and cfg_.get("substyles"):
        tempo = substyle_tempo(rows, ss.get(f"substyle_{style_name}", ""))
        if tempo:
            ss[f"bpm_{style_name}"] = tempo[0]
            if group not in defaults:
                ss[f"energy_{style_name}"] = tempo[1]


# ----------------------------------------------------------------- main UI
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Escala")
    keys = cfg["keys"]
    key = st.selectbox("Tónica", keys, index=keys.index("C"), key=f"key_{style}")
    mode = st.radio("Modo", ["major", "minor"], horizontal=True, key=f"mode_{style}",
                    format_func=lambda m: {"major": "Mayor", "minor": "Menor"}[m])
    variation = st.selectbox("Variación", cfg["variations"][mode], key=f"variation_{style}_{mode}")
with c2:
    st.subheader("Carácter")
    grouped = cfg["mood_groups"]
    mood_group = grouper(grouped) if grouped else (lambda m: None)
    moods = sorted(set(cfg["moods"]) | {p["mood"] for p in progression_library if p["mood"]},
                   key=lambda m: (mood_group(m) or "", m) if grouped else m)
    ss.setdefault(f"mood_{style}", cfg["default_mood"] if cfg["default_mood"] in moods else moods[0])
    substyle = None
    if cfg.get("substyles"):
        subs = substyle_options(meta_rows)
        if subs:
            default_sub = next((v for v in subs if v.lower() == cfg["default_substyle"]), subs[0])
            if ss.get(f"substyle_{style}") not in subs:
                ss[f"substyle_{style}"] = default_sub
            counts = {v: sum(1 for r in meta_rows if str(r.get("style", "")).strip().lower() == v.lower())
                      for v in subs}
            substyle = st.selectbox(
                "Subestilo", subs, key=f"substyle_{style}", on_change=suggest_bpm_energy,
                args=(style, meta_rows), format_func=lambda v: f"{v} ({counts[v]} canciones)",
                help="Se guarda en la columna style del Sheet. La progresión, los patrones y los ritmos de "
                     "melodía se eligen entre tus canciones de ese subestilo (si alguno falta, entre todas) "
                     "y el BPM sugerido es el de esas canciones.")
    if grouped:
        mood = st.selectbox("Mood", moods, key=f"mood_{style}", on_change=suggest_bpm_energy,
                            args=(style, meta_rows) if substyle else (style,),
                            format_func=lambda m: f"{m} ({GROUP_LABELS.get(mood_group(m), 'otro')})",
                            help="La progresión y los patrones se eligen entre las canciones del mismo grupo "
                                 f"({' / '.join(GROUP_LABELS.get(g, g) for g in grouped)}).")
    else:
        mood = st.selectbox("Mood", moods, key=f"mood_{style}",
                            help="Elige la progresión de acordes entre las canciones con ese mood.")
    length_bars = st.select_slider("Longitud (compases)", LENGTHS, value=16, key=f"len_{style}")
    temperature = st.slider("Temperatura", 0.3, 1.5, 0.9, 0.05, key=f"temp_{style}",
                            help="Más baja = más predecible y más riesgo de copiar la biblioteca. "
                                 "0.9–1.0 es un buen punto.")
with c3:
    st.subheader("Registro")
    tag_key = f"tag_input_{style}"
    if f"next_tag_{style}" in ss:
        ss[tag_key] = ss.pop(f"next_tag_{style}")
    ss.setdefault(tag_key, suggested_tag())
    tag_input = st.text_input("Nombre del registro", key=tag_key,
                              help="Formato song_021. Se sugiere el siguiente número libre de este estilo.")
    tag_num, tag_error = validate_tag(tag_input)
    prefix = cfg["file_prefix"]
    if tag_error:
        st.error(tag_error)
    else:
        shown = cfg["name_format"].format(n=tag_num)
        st.caption(f"Se guardará como **{shown}** · archivos `{prefix}melody_{tag_num:03d}.mid`, …")
    retry = st.checkbox("Reintentar solo si sale COPIA (hasta 10 veces)", value=True, key=f"retry_{style}")
    if cfg["bpm_range"]:
        lo_bpm, hi_bpm = cfg["bpm_range"]
        lo_en, hi_en = cfg["energy_range"]
        fallback = ((lo_bpm + hi_bpm) // 2, round((lo_en + hi_en) / 2, 2))
        g = mood_group(mood)
        sub_tempo = substyle_tempo(meta_rows, substyle) if substyle else None
        ss.setdefault(f"bpm_{style}", sub_tempo[0] if sub_tempo else cfg["group_defaults"].get(g, fallback)[0])
        ss.setdefault(f"energy_{style}", cfg["group_defaults"].get(g, fallback)[1])
        bpm = st.slider("BPM", lo_bpm, hi_bpm, key=f"bpm_{style}",
                        help=("Sugerido según el subestilo; puedes cambiarlo." if substyle
                              else "Sugerido según el mood; puedes cambiarlo."))
        energy = st.slider("Energy", lo_en, hi_en, step=0.05, key=f"energy_{style}",
                           help="Sugerida según el mood; puedes cambiarla.")
    else:
        bpm = energy = None
    st.caption(cfg["fixed_caption"])


def pattern_options(part):
    return ["random", patterns.BASIC] + [p["id"] for p in pattern_library.get(part, [])]


def pattern_label(part):
    density = {p["id"]: p["notes_per_bar"] for p in pattern_library.get(part, [])}
    basic = {"harmony": "Básico (acordes sostenidos)"}.get(part, "Básico (el patrón fijo original)")

    def fmt(opt):
        if opt == "random":
            return f"Aleatorio (entre {len(density)} patrones de la biblioteca)"
        if opt == patterns.BASIC:
            return basic
        return f"Estilo {opt} · {density[opt]:.0f} notas por compás"
    return fmt


st.subheader("Acompañamiento")
a1, a2, a3 = st.columns(3)
arp_choice = a1.selectbox("Patrón de arpegio", pattern_options("arpeggio"), format_func=pattern_label("arpeggio"),
                          key=f"arp_{style}",
                          help="Los patrones salen de los arpegios de las canciones de la biblioteca y se "
                               "adaptan a los acordes y la escala de la canción nueva.")
bass_choice = a2.selectbox("Patrón de bajo", pattern_options("bass"), format_func=pattern_label("bass"),
                           key=f"bass_{style}",
                           help="Igual que el arpegio, a partir de los bajos de la biblioteca.")
params = dict(key=key, mode=mode, variation=variation, mood=mood,
              length_bars=length_bars, temperature=temperature,
              arp_choice=arp_choice, bass_choice=bass_choice)
if substyle:
    params["substyle"] = substyle
if cfg["harmony"] == "patterns":
    params["harmony_choice"] = a3.selectbox(
        "Comping de armonía", pattern_options("harmony"), format_func=pattern_label("harmony"),
        key=f"harm_{style}",
        help=f"Ritmo y voicings de la armonía de tus canciones {cfg['label'].lower()}, re-armados "
             "sobre los acordes nuevos.")
if cfg["melody_center"] is not None:
    params["octave"] = a3.radio("Registro de la melodía", list(OCTAVES), index=1, horizontal=True,
                                format_func=OCTAVES.get, key=f"octave_{style}")
if cfg["bpm_range"]:
    params.update(bpm=bpm, energy=energy)
if cfg.get("melody_source") in ("templates", "hybrid"):
    hooks = [t for t in pattern_library.get("melody_templates", []) if t["kind"] == "hook"]
    use_model = cfg.get("melody_source") == "hybrid" and model is not None
    params["lead_choice"] = a1.selectbox(
        "Lead", (["model"] if use_model else []) + ["random"] + [t["id"] for t in hooks], key=f"lead_{style}",
        format_func=lambda o: ("Modelo entrenado (motivo de 4 compases que se repite)" if o == "model" else
                               f"Ritmo de un gancho tuyo, al azar (entre {len(hooks)})" if o == "random" else
                               f"Ritmo de {o} · {next(t['notes_per_bar'] for t in hooks if t['id'] == o):.1f} notas por compás"),
        help="Modelo: el LSTM entrenado con tus melodías techno escribe un motivo, que se ajusta a los acordes "
             "y se repite. Ritmo: el lead toma solo el RITMO de una de tus melodías; las notas son nuevas.")
if "arpeggio2" in cfg["parts"]:
    params["arp2_choice"] = a2.selectbox(
        "Arpegio 2", ["auto"] + [p["id"] for p in pattern_library.get("arpeggio", [])], key=f"arp2_{style}",
        format_func=lambda o: "Complementario (otro registro y densidad)" if o == "auto" else f"Estilo {o}",
        help="Segundo arpegio con otro papel: si el primero viene de una canción con dos arpegios (002), "
             "se usa su pareja; si no, el más distinto en registro y densidad.")

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("🎼 Generar", type="primary", width="stretch"):
    with st.spinner("Generando melodía y verificando originalidad…"):
        run_generation(params, retry)
if ss.song is not None and b2.button("🔄 Regenerar", width="stretch",
                                     help="Nueva melodía (y nuevos patrones, si están en Aleatorio) con los parámetros actuales."):
    with st.spinner("Regenerando…"):
        run_generation(params, retry)

song, report = ss.song, ss.report
if song is None:
    st.info("Elige la escala y los parámetros, y presiona **Generar**.")
    st.stop()

st.divider()

# ---- summary
changed = any(song.get(k) != v for k, v in params.items())
if changed:
    st.caption("⚠️ Cambiaste parámetros después de generar; lo que ves (y lo que se guardaría) es la canción generada "
               "con los parámetros de abajo. Presiona Regenerar para aplicar los nuevos.")
s1, s2 = st.columns([3, 2])
with s1:
    extra_title = f" · {song['bpm']:g} bpm" if cfg["bpm_range"] else ""
    if song.get("substyle"):
        extra_title = f" · {song['substyle']}" + extra_title
    st.subheader(f"{song['key']} {'mayor' if song['mode'] == 'major' else 'menor'} "
                 f"({song['variation']}) · {song['mood']} · {song['length_bars']} compases{extra_title}")
    st.markdown("**Acordes:** " + "  →  ".join(f"`{c}`" for c in song["chords"]))
    accomp = f"arpegio: {pattern_name(song['arp_pattern'])} · bajo: {pattern_name(song['bass_pattern'])}"
    if song.get("harmony_pattern"):
        accomp += f" · comping: {pattern_name(song['harmony_pattern'])}"
    if song.get("arp2_pattern"):
        accomp += f" · arpegio 2: {pattern_name(song['arp2_pattern'])}"
    if song.get("lead_template"):
        lead = "modelo" if song["lead_template"] == "model" else f"ritmo de {song['lead_template']}"
        accomp += f" · lead: {lead} · contramelodía: {song['counter_template']}"
    st.caption(f"Progresión {'-'.join(song['template_degrees'])} tomada de {song['template_source']} · "
               f"{accomp} · temperatura {song['temperature']} · semilla {song['seed']}"
               + (f" · {song['attempts']} intentos" if song.get("attempts", 1) > 1 else ""))
with s2:
    v = report["verdict"]
    which = "" if report.get("part", "melody") == "melody" else f" ({PART_LABELS[report['part']].lower()})"
    msg = f"{VERDICT_ICON[v]} **Melodía{which} {v}** — {report['reason']}"
    if v != "ORIGINAL" and report["closest"]:
        msg += f" (vs **{report['closest']}**)"
    {"ORIGINAL": st.success, "SIMILAR": st.warning, "COPIA": st.error}[v](msg)
    with st.expander("Detalle de similitud (5 más parecidas)"):
        det = pd.DataFrame([{
            "Canción": r["song"], "Veredicto": r["verdict"],
            "% coincidente": f"{r['coverage']:.0%}", "Frase más larga (notas)": r["run"], "Comparación": r["mode"],
        } for r in report["rows"][:5]])
        st.dataframe(det, hide_index=True, width="stretch")
        st.caption(f"{report['n_notes']} notas en la melodía. SIMILAR: ≥{originality.WARN_RUN} notas seguidas o "
                   f"≥{originality.WARN_COVERAGE:.0%}. COPIA: ≥{originality.COPY_RUN} notas seguidas o "
                   f"≥{originality.COPY_COVERAGE:.0%}.")

# ---- listen
st.subheader("Escuchar")
if "tracks" not in ss:
    with st.spinner("Sintetizando audio…"):
        ss.tracks = {p: synth.render_part(song["events"][p], song.get("bpm", pipeline.BPM), p, song["total_beats"])
                     for p in pipeline.song_parts(song)}
shown_parts = display_parts(song)
vcols = st.columns(len(shown_parts))
volumes = {}
song_defaults = STYLES[song.get("style", "ambient")]["default_volumes"]
for col, part in zip(vcols, shown_parts):
    volumes[part] = col.slider(f"Volumen {PART_LABELS[part].lower()}", 0.0, 1.0,
                               song_defaults.get(part, 0.5), 0.05, key=f"vol_{style}_{part}")
if any(volumes.values()):
    st.audio(synth.mix_to_wav(ss.tracks, volumes), format="audio/wav")
else:
    st.caption("Todas las pistas están en silencio.")
st.altair_chart(piano_roll(song), width="stretch")

# ---- save
st.divider()
st.subheader("Guardar")
saved = ss.saved
if saved:
    st.success(f"Registrado como **{saved['name']}** en la fila {saved['row']} de la pestaña {cfg['label'].lower()}."
               + ("" if saved.get("melody_error") else
                  f" La melodía quedó guardada en la pestaña «{cfg['melody_tab']}» para futuras comparaciones."))
    if saved.get("melody_error"):
        st.warning(f"La fila se registró, pero no se pudo guardar la melodía en la pestaña «{cfg['melody_tab']}»: "
                   f"{saved['melody_error']}. Sube el zip en la barra lateral para guardarla.")
    st.download_button(f"⬇️ Descargar song_{saved['tag']}.zip", saved["zip"],
                       file_name=f"song_{saved['tag']}.zip", mime="application/zip", type="primary")
    st.caption(f"Descomprime el zip dentro de tu carpeta {cfg['folder_hint']}.")
else:
    blockers = []
    if report["verdict"] == "COPIA":
        blockers.append("la melodía es una copia; regenera")
    if tag_error:
        blockers.append(tag_error)
    if ws is None:
        blockers.append("no hay conexión al Google Sheet")
    if report["verdict"] == "SIMILAR":
        st.warning("La melodía se parece bastante a una existente. Puedes guardarla igual.")
    if blockers:
        st.caption("No se puede guardar: " + "; ".join(blockers) + ".")
    if st.button("💾 Guardar y registrar", type="primary", disabled=bool(blockers)):
        tag = f"{tag_num:03d}"
        record = pipeline.sheet_record(song, tag_num, tag)
        try:
            row = sheets.append_song(ws, record)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo escribir en el Sheet: {e}")
        else:
            ss.saved = {"tag": tag, "row": row, "name": record["Name"], "zip": pipeline.song_zip(song, tag)}
            for part in styles.melody_parts(cfg):  # lead (and counter-line in techno)
                mtag = tag if part == "melody" else f"{tag}/{part}"
                try:
                    sheets.save_melody(ws, mtag, song["key"], song["mode"], song["variation"],
                                       song["events"][part], now_str(), tab=cfg["melody_tab"])
                except Exception as e:  # noqa: BLE001
                    ss.saved["melody_error"] = str(e)
                ss.saved_songs.setdefault(style, {})[f"song_{mtag}"] = originality.make_entry(
                    f"song_{mtag}", song["events"][part], song["scale"])
            load_sheet_melodies.clear()
            ss[f"next_tag_{style}"] = f"song_{tag_num + 1:03d}"
            load_metadata.clear()
            st.rerun()
