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
