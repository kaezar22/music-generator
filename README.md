# MusicAI — generador de canciones (Streamlit)

Genera una canción nueva: melodía con el LSTM del proyecto, más arpegio, bajo y armonía a partir de una progresión de acordes. Verifica que la melodía no copie las canciones existentes, la deja escuchar y la registra en el Google Sheet. Al final entrega un `.zip` con la carpeta `song_XXX`.

Hay dos estilos, que se eligen arriba en la app: **Ambient** (el original) y **Lo-fi** (entrenado con las 20 canciones lo-fi compuestas a mano). Ver [Modo lo-fi](#modo-lo-fi).

## Flujo en la app

1. Entra con la contraseña.
2. Elige:
   - **tónica, modo y variación** (la escala);
   - **mood**, que decide la progresión de acordes;
   - **longitud**: 8, 16, 32 o 64 compases;
   - **temperatura**.
3. **Generar.** La melodía se compara contra la biblioteca:
   - **ORIGINAL:** se puede guardar.
   - **SIMILAR:** se muestra una advertencia, pero se puede guardar.
   - **COPIA:** el botón de guardar queda bloqueado. Usa **Regenerar**, o deja marcado "Reintentar solo si sale COPIA".
4. Escucha la mezcla (con volumen por pista) y revisa el piano-roll.
5. **Guardar y registrar.** Escribe la fila en el Sheet, guarda la melodía en la pestaña `melodies` y habilita la descarga de `song_XXX.zip`. Descomprímelo dentro de `AI_MUSIC_Project`.

Valores fijos: 60 bpm, 4/4, energy 0.2, style ambient (están en `musicgen/pipeline.py`).

## Modo lo-fi

Al elegir **Lo-fi** cambian estas cosas:

- **Modelo de melodía:** `models/lofi/` (entrenado con `tools/train_melody.py`). La melodía se centra alrededor de C5; el selector *Registro de la melodía* la baja o la sube una octava.
- **Escala:** variaciones natural, dorian, lydian, mixolydian y harmonic. En menor armónica la 7ª elevada solo aparece sobre los acordes dominantes, como en las canciones a mano.
- **Acordes extendidos:** la progresión sale de una canción lo-fi del mismo grupo de mood y se reconstruye en la escala nueva con el mismo tamaño de acorde (maj9, m9, 13, 7b9…). Si el comping toca una tensión extra, el nombre del acorde la incluye, así la columna `chords` coincide con `harmony.mid`.
- **Acompañamiento:** patrones de bajo, arpegio y **comping de armonía** sacados de `library_lofi/` (las 20 canciones lo-fi).
- **Mood:** las 15 etiquetas, agrupadas en calma / melancólico / luminoso (`musicgen/moods.py`). El grupo elige progresión y sugiere BPM y energy.
- **BPM (70–90) y energy (0.25–0.6):** los eliges tú; la app sugiere valores según el mood.
- **Swing:** no se aplica. Los MIDI salen rectos para que también sirvan para reentrenar; el swing lo pones en el DAW.
- **Guardado:**
  - fila en la pestaña lo-fi del Sheet (gid 913708363) con `Name` tipo `song021` y archivos `lofi_melody_021.mid`, …;
  - melodía en la pestaña `melodies lofi`;
  - zip con la carpeta `song_021`: la melodía con su longitud completa, y armonía, bajo y arpegio como loop de 4 compases (igual que el dataset). Descomprímelo dentro de `lofi_songs`.
- **Originalidad:** cada melodía lo-fi se compara contra las lo-fi y también contra las ambient.

Las canciones que genera la app en lo-fi pasan `tools/validate_lofi_dataset.py`, así que pueden sumarse al dataset para un futuro reentrenamiento.

Secret opcional: `lofi_worksheet_gid` si la pestaña lo-fi cambia de gid (por defecto 913708363).

## Modo melodic techno

Estilo `worakls` (pestaña del Sheet gid 297000043; secret opcional `techno_worksheet_gid`). Genera 6 partes: lead, contramelodía, 2 arpegios, bajo y armonía. Biblioteca: `library_techno/` (20 canciones + `metadata.csv` exportado de la pestaña). 120–140 BPM, energy 0.6–0.95.

- **Acompañamiento:** patrones de bajo, arpegio y armonía sacados de tus canciones, tomando los archivos de las columnas del Sheet (`bass` + `bass2`, `arpeggio` + `arpeggio2`, `harmony` + `harmony2`). Acepta pads que suenan 2 compases, loops de 8 y los cromatismos que escribiste (G# bajo A en el bajo de 001, C# sobre F en el arpegio 2 de 002). Un bajo o una armonía con menos del 70 % de notas de los acordes de su canción no se usa (009 bass2/harmony2 y 010 harmony están sobre otra progresión).
- **Acordes:** la progresión conserva la calidad original de cada acorde (v menor y V mayor en la misma canción). Si una calidad no encaja en la tonalidad nueva, el acorde se reconstruye con la escala. Escalas: menor natural, armónica, dórica, frigia y mayor.
- **Lead (selector "Lead"):**
  - *Modelo entrenado* (por defecto): el LSTM de `models/techno/` escribe un motivo de 4 compases y una respuesta; las notas de tiempos fuertes y las largas van a notas del acorde, y el motivo se repite sobre la progresión.
  - *Ritmo de un gancho tuyo*: toma solo el ritmo de una de tus melodías rítmicas; las notas son nuevas.
- **Contramelodía:** por reglas, con el ritmo de una de tus líneas lentas variado. Va por debajo del lead, con notas del acorde y el menor movimiento posible, buscando terceras y sextas, y sin unísonos ni quintas u octavas paralelas.
- **Arpegios en pareja:** si el primero viene de una canción con dos arpegios se usa su pareja; si no, el más distinto en registro y densidad (y se separan una octava si quedan juntos).
- **Originalidad:** el lead y la contramelodía se comparan contra tus 34 melodías techno; las dos se guardan en la pestaña `melodies techno`.
- **Guardado:** fila `song_XXX` con `style = worakls` y una columna por parte; zip con los archivos `techno_…_XXX.mid` (melodías completas, acompañamiento como loop).

### Reentrenar el modelo techno

Copia las canciones nuevas a `library_techno/song_XXX/`, exporta la pestaña del Sheet a `library_techno/metadata.csv` y corre (necesita torch):

    python tools/train_techno.py

Aparta 3 canciones para validar, compara "solo techno" contra "preentrenado con ambient + lo-fi y afinado con techno", y exporta la ganadora reentrenada con todo a `models/techno/` (con `training_report.json`). No toca los modelos ambient ni lo-fi.

## Biblioteca

La app compara cada melodía nueva contra dos fuentes:

- **`library/`:** las 20 canciones originales, incluidas en el repo.
- **Pestaña `melodies` del Google Sheet:** cada vez que presionas **Guardar y registrar**, la app guarda ahí las notas de la melodía como texto. Al arrancar las lee, así que la biblioteca crece sola, sin commits ni uploads. La pestaña se crea sola la primera vez. No la edites a mano.

Si una canción está registrada en el Sheet pero su melodía no está en ninguna de las dos fuentes, la barra lateral lo avisa. Para arreglarlo, sube su zip (o tu carpeta `AI_MUSIC_Project` comprimida) en **Agregar canciones** y presiona **Guardar N melodía(s) en el Sheet**. Solo hace falta una vez por canción.

La escala de cada canción se lee del Google Sheet, así que la comparación por grados de la escala funciona también con las canciones nuevas.

## Patrones de arpegio y bajo

El arpegio y el bajo ya no usan un patrón fijo. La app extrae los patrones de los archivos `arpeggio_XXX.mid` y `bass_XXX.mid` de `library/` (19 de arpegio y 13 de bajo distintos) y los adapta a los acordes y la escala de la canción nueva. Cada nota se guarda como su posición en el compás y su grado dentro del acorde (fundamental, tercera, quinta o nota de paso).

En **Acompañamiento** puedes dejar cada uno en *Aleatorio*, elegir el estilo de una canción concreta, o volver al patrón *Básico* original. Si agregas carpetas `song_XXX` a `library/` (con su fila en `library/metadata.csv`), sus patrones se suman solos. La lógica está en `musicgen/patterns.py`.

## Correr localmente

```bash
cd musicai_webapp
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python make_secrets.py ..\AI_MUSIC_Project\credentials.json   # crea .streamlit/secrets.toml
streamlit run app.py
```

## Publicar en Streamlit Cloud

1. Crea un repositorio en GitHub (puede ser privado) y sube esta carpeta:
   ```bash
   git init
   git add .
   git commit -m "MusicAI webapp"
   git branch -M main
   git remote add origin https://github.com/<tu-usuario>/musicai_webapp.git
   git push -u origin main
   ```
   Antes del commit, revisa con `git status` que **no** aparezcan `credentials.json` ni `.streamlit/secrets.toml`. El `.gitignore` los excluye.
2. En https://share.streamlit.io, entra en **Create app**, elige el repo, la rama `main` y el archivo `app.py`. En *Advanced settings* elige Python 3.12.
3. En **Advanced settings → Secrets**, pega el contenido de tu `.streamlit/secrets.toml` local, el que generó `make_secrets.py`.
4. **Deploy.**

La cuenta de servicio (`musicai-api@musicai-505201.iam.gserviceaccount.com`) necesita permiso de **Editor** en el Sheet.

## Si vuelves a entrenar el modelo

La app no usa PyTorch: el LSTM está reimplementado en NumPy (`musicgen/melody.py`) y da los mismos resultados. Después de reentrenar en el notebook:

```bash
python tools/export_weights.py ..\AI_MUSIC_Project\melody_lstm.pt
copy ..\AI_MUSIC_Project\melody_vocab.json models\
```

## Estructura

```
app.py                      interfaz Streamlit
musicgen/theory.py          escalas, acordes, progresiones, bajo/arpegio/armonía
musicgen/melody.py          LSTM en NumPy + decodificación a notas
musicgen/patterns.py        patrones de arpegio, bajo y comping aprendidos de library/ y library_lofi/
musicgen/moods.py           grupos de mood por estilo
musicgen/styles.py          todas las diferencias entre estilos (ambient, lofi, worakls)
musicgen/techno_melody.py   lead (modelo o plantilla rítmica) y contramelodía por reglas (techno)
musicgen/originality.py     verificación de similitud de la melodía
musicgen/synth.py           sintetizador para escuchar en el navegador
musicgen/midi_io.py         lectura/escritura MIDI en memoria
musicgen/sheets.py          lectura/escritura del Google Sheet
musicgen/pipeline.py        generación completa, zip y fila del Sheet
models/                     pesos del LSTM ambient (.npz) y vocabulario
models/lofi/                modelo lo-fi + training_report.json
models/techno/              modelo melodic techno + training_report.json
library/                    canciones ambient + metadata.csv
library_lofi/               las 20 canciones lo-fi a mano + metadata.csv
library_techno/             las 20 canciones melodic techno + metadata.csv
tools/validate_lofi_dataset.py   valida el dataset lo-fi (Sheet + MIDI) antes de entrenar
tools/train_melody.py            entrena y exporta el modelo lo-fi
tools/train_techno.py            entrena y exporta el modelo melodic techno
```

## Umbrales de similitud

En `musicgen/originality.py`:

| Veredicto | Condición |
|---|---|
| **COPIA** | idéntica, o transpuesta, o con cambio de modo |
| | ≥ 60 % de la melodía coincide con una canción |
| | contiene la mitad de una melodía existente |
| | la mitad de la nueva es una sola cita |
| | ≥ 24 notas seguidas iguales y ≥ 30 % coincidente |
| **SIMILAR** | ≥ 12 notas seguidas iguales, o ≥ 30 % coincidente |
| **ORIGINAL** | lo demás |

Entre las 20 canciones actuales la coincidencia máxima es de 6 notas, así que todas salen ORIGINAL entre sí.
