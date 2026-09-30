"""Google Sheet access (song metadata) through the service account."""

import re

import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def open_worksheet(service_account_info, spreadsheet_id, worksheet_gid):
    creds = Credentials.from_service_account_info(dict(service_account_info), scopes=SCOPES)
    gc = gspread.authorize(creds)
    return gc.open_by_key(spreadsheet_id).get_worksheet_by_id(int(worksheet_gid))


def read_metadata(ws):
    """Rows as dicts (header stripped), skipping rows without a Name."""
    values = ws.get_all_values()
    if not values:
        return []
    header = [h.strip() for h in values[0]]
    rows = []
    for raw in values[1:]:
        row = {h: (raw[i].strip() if i < len(raw) else "") for i, h in enumerate(header) if h}
        if row.get("Name"):
            rows.append(row)
    return rows


def song_number(name):
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


def append_song(ws, record):
    """Write record (dict keyed by header names) into the first row whose Name
    cell is empty — the sheet has blank formatted rows at the bottom, so a plain
    append would land after them. Returns the 1-based row number written."""
    values = ws.get_all_values()
    header = [h.strip() for h in values[0]]
    target = None
    for i, raw in enumerate(values[1:], start=2):
        if not (raw[0].strip() if raw else ""):
            target = i
            break
    if target is None:
        target = len(values) + 1
    row = [record.get(h, "") for h in header]
    end_col = gspread.utils.rowcol_to_a1(target, len(header))
    # RAW so that "4/4" stays text instead of turning into a date
    ws.update(range_name=f"A{target}:{end_col}", values=[row], value_input_option="RAW")
    return target


# ------------------------------------------------------------------ melodies tab
# The app server does not keep files, so each saved melody is also stored as
# text in a separate tab of the same spreadsheet. On start-up the app rebuilds
# the comparison library from this tab — no commits or uploads needed.
MELODY_TAB = "melodies"
MELODY_HEADER = ["song", "key", "mode", "variation", "n_notes", "notes", "saved_at"]
GRID = 4  # 16th notes per beat


def encode_notes(notes):
    """[(onset_beats, pitch, dur_beats)] -> 'onset:pitch:dur;...' in 16ths."""
    return ";".join(f"{round(o * GRID)}:{p}:{max(1, round(d * GRID))}" for o, p, d in notes)


def decode_notes(text):
    notes = []
    for item in str(text).split(";"):
        parts = item.strip().split(":")
        if len(parts) != 3:
            continue
        o, p, d = (int(x) for x in parts)
        notes.append((o / GRID, p, d / GRID))
    return notes


def melody_worksheet(main_ws, create=True):
    """The 'melodies' tab next to the metadata tab (created on first use)."""
    sh = main_ws.spreadsheet
    try:
        return sh.worksheet(MELODY_TAB)
    except gspread.WorksheetNotFound:
        if not create:
            return None
        ws = sh.add_worksheet(title=MELODY_TAB, rows=200, cols=len(MELODY_HEADER))
        ws.update(range_name="A1", values=[MELODY_HEADER], value_input_option="RAW")
        return ws


def read_melodies(main_ws):
    """[{song, key, mode, variation, notes}] stored in the melodies tab."""
    ws = melody_worksheet(main_ws, create=False)
    if ws is None:
        return []
    values = ws.get_all_values()
    if not values:
        return []
    header = [h.strip() for h in values[0]]
    out = []
    for raw in values[1:]:
        row = {h: (raw[i].strip() if i < len(raw) else "") for i, h in enumerate(header)}
        if row.get("song") and row.get("notes"):
            row["notes"] = decode_notes(row["notes"])
            out.append(row)
    return out


def save_melody(main_ws, tag, key, mode, variation, notes, saved_at):
    """Append one melody (tag like '021'); replaces an existing row for the same song."""
    ws = melody_worksheet(main_ws, create=True)
    song = f"song_{tag}"
    row = [song, str(key).lower(), mode, variation, len(notes), encode_notes(notes), saved_at]
    existing = ws.col_values(1)
    if song in existing:
        r = existing.index(song) + 1
        ws.update(range_name=f"A{r}", values=[row], value_input_option="RAW")
    else:
        ws.append_row(row, value_input_option="RAW", table_range="A1")
    return song
