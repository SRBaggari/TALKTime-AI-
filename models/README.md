# Models

TalkTime AI loads models from the standard Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub`), so this folder is normally empty.

| Model | Purpose | Size | Licence / access |
|---|---|---|---|
| `Systran/faster-whisper-small` (default) | Transcription | ~480 MB | MIT, public |
| `Systran/faster-whisper-{tiny,base,medium}` | Alternative sizes | 75 MB – 1.5 GB | MIT, public |
| `pyannote/speaker-diarization-community-1` | Speaker diarization | ~30 MB | CC-BY-4.0, gated (accept conditions once) |
| `en_core_web_sm` (spaCy) | Topics / noun phrases | ~13 MB | MIT, installed as a Python package |

Download with `scripts\download_models.py` (see the main README).

To use a manually copied diarization pipeline, place the folder (with
`config.yaml`) anywhere and set `TALKTIME_DIARIZATION_PATH` in `.env`.
