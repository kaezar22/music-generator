"""MusicAI — generador de canciones (melodía LSTM + arpegio + bajo + armonía).

Flujo: elegir escala / mood / longitud / temperatura -> generar -> verificar
originalidad de la melodía -> escuchar -> guardar (fila en Google Sheet + zip).
"""

import csv
import os
import re
import secrets as pysecrets
from datetime import datetime
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from musicgen import originality, pipeline, sheets, synth, theory
from musicgen.melody import MelodyModel

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LIBRARY_DIR = os.path.join(APP_DIR, "library")
LOCAL_METADATA = os.path.join(LIBRARY_DIR, "metadata.csv")

KEYS = theory.NOTE_NAMES
DEFAULT_MOODS = ["quiet", "melancolic", "cinematic", "optimistic", "flamenco"]
LENGTHS = [8, 16, 32, 64]
PART_LABELS = {"melody": "Melodía", "arpeggio": "Arpegio", "bass": "Bajo", "harmony": "Armonía"}
PART_COLORS = {"melody": "#2a78d6", "arpeggio": "#eb6834", "bass": "#1baf7a", "harmony": "#a3a29c"}
DEFAULT_VOLUMES = {"melody": 1.0, "arpeggio": 0.45, "bass": 0.6, "harmony": 0.0}
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


# ----------------------------------------------------------------- resources
@st.cache_resource
def get_model():
    return MelodyModel()


@st.cache_resource
def get_worksheet():
    if "gcp_service_account" not in st.secrets:
        return None
    return sheets.open_worksheet(
        st.secrets["gcp_service_account"],
        st.secrets.get("spreadsheet_id", "1z_njzCTDkI59D6dggXbFxONousXaHWe1wGoilxFo-Pc"),
        st.secrets.get("worksheet_gid", 1183065346),
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_metadata():
    """(rows, source). Sheet first; bundled CSV as fallback."""
    try:
        ws = get_worksheet()
        if ws is not None:
            return sheets.read_metadata(ws), "Google Sheet"
    except Exception as e:  # noqa: BLE001
        st.session_state.sheet_error = str(e)
    with open(LOCAL_METADATA, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [r for r in rows if r.get("Name")], "CSV incluido en la app"


@st.cache_data(ttl=300, show_spinner=False)
def load_sheet_melodies():
    """Melodies saved by the app in the 'melodies' tab: (rows, error)."""
    try:
        ws = get_worksheet()
        if ws is None:
            return [], None
        return sheets.read_melodies(ws), None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


def metadata_by_num(rows):
    out = {}
    for r in rows:
        n = sheets.song_number(r.get("Name"))
        if n is not None:
            out[n] = r
    return out


@st.cache_data(show_spinner=False)
def bundled_library(meta_rows):
    return originality.load_folder_library(LIBRARY_DIR, metadata_by_num(meta_rows))


@st.cache_data(show_spinner=False)
def zip_library(zip_bytes, meta_rows):
    return originality.load_zip_library(zip_bytes, metadata_by_num(meta_rows))


# ----------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("saved_songs", {})   # tag -> entry (songs saved in this session)
ss.setdefault("song", None)
ss.setdefault("report", None)
ss.setdefault("saved", None)       # {"tag","zip","row"} for the current song

meta_rows, meta_source = load_metadata()
progression_library = theory.build_progression_library(meta_rows)
model = get_model()

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Biblioteca")
    library = dict(bundled_library(meta_rows))
    n_bundled = len(library)
    sheet_melodies, mel_error = load_sheet_melodies()
    in_sheet_tab = set()
    for r in sheet_melodies:
        in_sheet_tab.add(r["song"])
        if r["song"] in library:
            continue
        try:
            scale = theory.scale_notes_ordered(r["key"], r["mode"], r.get("variation", "natural"))
        except Exception:  # noqa: BLE001
            scale = None
        library[r["song"]] = originality.make_entry(r["song"], r["notes"], scale)
    st.caption(f"{n_bundled} incluidas en la app · {len(library) - n_bundled} guardadas en el Sheet.")
    if mel_error:
        st.error(f"No se pudo leer la pestaña de melodías: {mel_error}")
    up = st.file_uploader(
        "Agregar canciones (.zip)", type=["zip"],
        help="Comprime tus carpetas song_XXX (o la carpeta AI_MUSIC_Project completa) y súbela. "
             "Se usa para comparar melodías y para no repetir nombres de registro.",
    )
    if up is not None:
        extra = zip_library(up.getvalue(), meta_rows)
        new = [k for k in extra if k not in library]
        library.update(extra)
        st.caption(f"Zip: {len(extra)} melodías leídas ({len(new)} nuevas).")
        # melodies from the zip that are not stored yet -> offer to keep them in the Sheet
        to_store = sorted(k for k in new if k not in in_sheet_tab)
        if to_store and get_worksheet() is not None:
            if st.button(f"Guardar {len(to_store)} melodía(s) en el Sheet", type="primary",
                         help="Así quedan en la biblioteca de forma permanente: " + ", ".join(to_store)):
                meta_num = metadata_by_num(meta_rows)
                saved_now = []
                for k in to_store:
                    num = int(k.split("_")[1])
                    row = meta_num.get(num, {})
                    sheets.save_melody(get_worksheet(), f"{num:03d}", row.get("Key", ""), row.get("mode", ""),
                                       row.get("variation", ""), extra[k]["notes"], now_str())
                    saved_now.append(k)
                load_sheet_melodies.clear()
                st.success("Guardadas: " + ", ".join(saved_now))
                st.rerun()
    library.update(ss.saved_songs)
    st.metric("Melodías para comparar", len(library))
    registered = {f"song_{n:03d}" for n in metadata_by_num(meta_rows)}
    missing = sorted(registered - set(library))
    if missing:
        st.warning("Registradas en el Sheet pero sin melodía para comparar: " + ", ".join(missing)
                   + ". Sube sus zips arriba y presiona «Guardar … en el Sheet».")

    st.divider()
    st.header("Google Sheet")
    ws = None
    try:
        ws = get_worksheet()
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo abrir el Sheet: {e}")
    if ws is None:
        st.warning("Sin conexión al Sheet: configura `gcp_service_account` en los secrets. "
                   "Se usa el CSV incluido como metadata.")
    else:
        st.caption(f"Conectado · {len(meta_rows)} registros leídos de: {meta_source}")
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
    nums |= {int(k.split("_")[1]) for k in library}
    return {n for n in nums if n is not None}


def suggested_tag():
    return f"song_{max(used_numbers(), default=0) + 1:03d}"


def validate_tag(tag):
    m = re.fullmatch(r"song_(\d{3,})", tag.strip())
    if not m:
        return None, "El nombre debe tener el formato song_021."
    num = int(m.group(1))
    if num in used_numbers():
        return None, f"{tag} ya existe en el Sheet o en la biblioteca."
    return num, None


def run_generation(params, retry_until_ok, max_tries=10):
    tries = max_tries if retry_until_ok else 1
    for attempt in range(1, tries + 1):
        seed = pysecrets.randbits(32)
        song = pipeline.generate_song(model, progression_library, seed=seed, **params)
        report = originality.check_melody(song["events"]["melody"], song["scale"], library)
        if report["verdict"] != "COPIA":
            break
    song["attempts"] = attempt
    ss.song, ss.report, ss.saved = song, report, None
    ss.pop("tracks", None)


def piano_roll(song):
    rows = []
    for part in ("harmony", "bass", "arpeggio", "melody"):
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
    order = [PART_LABELS[p] for p in ("melody", "arpeggio", "bass", "harmony")]
    color = alt.Color("Parte:N", scale=alt.Scale(domain=order, range=[PART_COLORS[p] for p in
                      ("melody", "arpeggio", "bass", "harmony")]), legend=alt.Legend(orient="top", title=None))
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


# ----------------------------------------------------------------- main UI
st.title("🎵 MusicAI — generador de canciones")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Escala")
    key = st.selectbox("Tónica", KEYS, index=KEYS.index("C"))
    mode = st.radio("Modo", ["major", "minor"], horizontal=True,
                    format_func=lambda m: {"major": "Mayor", "minor": "Menor"}[m])
    variation = st.selectbox("Variación", theory.VARIATIONS[mode])
with c2:
    st.subheader("Carácter")
    moods = sorted(set(DEFAULT_MOODS) | {p["mood"] for p in progression_library if p["mood"]})
    mood = st.selectbox("Mood", moods, index=moods.index("quiet") if "quiet" in moods else 0,
                        help="Elige la progresión de acordes entre las canciones con ese mood.")
    length_bars = st.select_slider("Longitud (compases)", LENGTHS, value=16)
    temperature = st.slider("Temperatura", 0.3, 1.5, 0.9, 0.05,
                            help="Más baja = más predecible y más riesgo de copiar la biblioteca. "
                                 "Con 0.7 cerca de la mitad de las melodías salían copiadas; 0.9–1.0 es un buen punto.")
with c3:
    st.subheader("Registro")
    if "next_tag" in ss:
        ss.tag_input = ss.pop("next_tag")
    ss.setdefault("tag_input", suggested_tag())
    tag_input = st.text_input("Nombre del registro", key="tag_input",
                              help="Formato song_021. Se sugiere el siguiente número libre.")
    tag_num, tag_error = validate_tag(tag_input)
    if tag_error:
        st.error(tag_error)
    else:
        st.caption(f"Se guardará como **song {tag_num}** · archivos `melody_{tag_num:03d}.mid`, …")
    retry = st.checkbox("Reintentar solo si sale COPIA (hasta 10 veces)", value=True)
    st.caption(f"Fijos: {pipeline.BPM} bpm · {pipeline.TIME_SIG} · energy {pipeline.ENERGY} · {pipeline.STYLE}")

params = dict(key=key, mode=mode, variation=variation, mood=mood,
              length_bars=length_bars, temperature=temperature)

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("🎼 Generar", type="primary", width="stretch"):
    with st.spinner("Generando melodía y verificando originalidad…"):
        run_generation(params, retry)
if ss.song is not None and b2.button("🔄 Regenerar", width="stretch",
                                     help="Nueva melodía con los parámetros actuales."):
    with st.spinner("Regenerando…"):
        run_generation(params, retry)

song, report = ss.song, ss.report
if song is None:
    st.info("Elige la escala y los parámetros, y presiona **Generar**.")
    st.stop()

st.divider()

# ---- summary
changed = any(song[k] != v for k, v in params.items())
if changed:
    st.caption("⚠️ Cambiaste parámetros después de generar; lo que ves (y lo que se guardaría) es la canción generada "
               "con los parámetros de abajo. Presiona Regenerar para aplicar los nuevos.")
s1, s2 = st.columns([3, 2])
with s1:
    st.subheader(f"{song['key']} {'mayor' if song['mode'] == 'major' else 'menor'} "
                 f"({song['variation']}) · {song['mood']} · {song['length_bars']} compases")
    st.markdown("**Acordes:** " + "  →  ".join(f"`{c}`" for c in song["chords"])
                + f"  \n<small>Progresión {'-'.join(song['template_degrees'])} tomada de {song['template_source']} · "
                  f"temperatura {song['temperature']} · semilla {song['seed']}"
                + (f" · {song['attempts']} intentos" if song.get("attempts", 1) > 1 else "") + "</small>",
                unsafe_allow_html=True)
with s2:
    v = report["verdict"]
    msg = f"{VERDICT_ICON[v]} **Melodía {v}** — {report['reason']}"
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
        ss.tracks = {p: synth.render_part(song["events"][p], pipeline.BPM, p, song["total_beats"])
                     for p in pipeline.PARTS}
vcols = st.columns(4)
volumes = {}
for col, part in zip(vcols, ("melody", "arpeggio", "bass", "harmony")):
    volumes[part] = col.slider(f"Volumen {PART_LABELS[part].lower()}", 0.0, 1.0,
                               DEFAULT_VOLUMES[part], 0.05, key=f"vol_{part}")
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
    st.success(f"Registrado como **song {int(saved['tag'])}** en la fila {saved['row']} del Sheet."
               + ("" if saved.get("melody_error") else
                  " La melodía quedó guardada en la pestaña «melodies» para futuras comparaciones."))
    if saved.get("melody_error"):
        st.warning(f"La fila se registró, pero no se pudo guardar la melodía en la pestaña «melodies»: "
                   f"{saved['melody_error']}. Sube el zip en la barra lateral para guardarla.")
    st.download_button(f"⬇️ Descargar song_{saved['tag']}.zip", saved["zip"],
                       file_name=f"song_{saved['tag']}.zip", mime="application/zip", type="primary")
    st.caption("Descomprime el zip dentro de tu carpeta AI_MUSIC_Project.")
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
            ss.saved = {"tag": tag, "row": row, "zip": pipeline.song_zip(song, tag)}
            try:
                sheets.save_melody(ws, tag, song["key"], song["mode"], song["variation"],
                                   song["events"]["melody"], now_str())
            except Exception as e:  # noqa: BLE001
                ss.saved["melody_error"] = str(e)
            load_sheet_melodies.clear()
            ss.saved_songs[f"song_{tag}"] = originality.make_entry(
                f"song_{tag}", song["events"]["melody"], song["scale"])
            ss.next_tag = f"song_{tag_num + 1:03d}"
            load_metadata.clear()
            st.rerun()
