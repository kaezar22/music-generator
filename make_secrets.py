"""Crea .streamlit/secrets.toml a partir de credentials.json.

Uso:
    python make_secrets.py ..\\AI_MUSIC_Project\\credentials.json

Luego copia el contenido del archivo generado en Streamlit Cloud
(Settings -> Secrets). El archivo está en .gitignore: no se sube a GitHub.
"""
import json
import os
import sys

APP_PASSWORD = "2209"
SPREADSHEET_ID = "1z_njzCTDkI59D6dggXbFxONousXaHWe1wGoilxFo-Pc"
WORKSHEET_GID = 1183065346


def toml_str(v):
    return json.dumps(v)  # JSON string escaping is valid TOML basic-string escaping


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "AI_MUSIC_Project", "credentials.json")
    with open(src, encoding="utf-8") as f:
        creds = json.load(f)
    lines = [
        f"app_password = {toml_str(APP_PASSWORD)}",
        f"spreadsheet_id = {toml_str(SPREADSHEET_ID)}",
        f"worksheet_gid = {WORKSHEET_GID}",
        "",
        "[gcp_service_account]",
    ]
    lines += [f"{k} = {toml_str(v)}" for k, v in creds.items()]
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, ".streamlit", "secrets.toml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Escrito {out}  (cuenta: {creds.get('client_email')})")


if __name__ == "__main__":
    main()
