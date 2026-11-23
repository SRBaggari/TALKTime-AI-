# TalkTime AI

**Intelligent speaker diarization and conversation analytics — fully offline.**

Upload a call, interview or meeting recording. TalkTime AI identifies who spoke and when, measures each
speaker's talk time exactly, transcribes the conversation, and produces timelines, conversation insights and
PDF/CSV reports. All of this runs on your own computer: no paid APIs, no cloud transcription, and no recording
ever leaves the machine.

---

## Features

| Area | What you get | How it is produced |
|---|---|---|
| Speaker diarization | Automatic speaker count (or exact/min/max), start–end times, overlapping speech kept | pyannote `speaker-diarization-community-1` (local) |
| Transcription | Automatic language detection, word timestamps, punctuation, `[unclear]` flags | Faster-Whisper (CTranslate2, CPU int8) |
| Speaker-wise transcript | Each line assigned by strongest timestamp overlap, at word level; uncertain lines flagged | Interval alignment |
| Speaking time | Total, share of speech, share of recording, turns, average/longest/shortest turn, first/last spoke | Exact interval math |
| Overlaps | Overlap events, *possible* interruptions, backchannels, simultaneous starts | Timing heuristics, conservative wording |
| Silence | Total/longest/average silence, gaps between speakers, pauses within a speaker, response delays | Speech-activity gaps + acoustic energy check |
| Participation | Balance score (normalised entropy), dominant speaker, neutral imbalance notes | Documented formula |
| Questions | Questions per speaker, likely responses, possibly unanswered questions, confidence | Heuristics |
| Sentiment | Positive / neutral / negative per segment and speaker, trend over time (English) | VADER lexicon (text only) |
| Conversation intelligence | Topics, key points, decisions, action items, commitments, important statements | spaCy + rule-based extraction |
| Optional LLM summary | Abstractive summary | Your local Ollama (disabled by default, localhost only) |
| Timeline | Speaker lanes, overlap and silence lanes, click to play, play an individual segment | Plotly + synced audio |
| Corrections | Rename speakers, merge split speakers, reassign a transcript line; analytics recalculated | No re-transcription needed |
| Comparison | Duration, balance, turn length, interruptions/min, silence, sentiment across recordings | Speakers never matched across calls |
| Reports | PDF with charts, transcript and a limitations section; CSV and JSON exports | ReportLab, pandas |
| History | All analyses stored in SQLite; delete one recording or everything | SQLAlchemy |

Every on-screen section is tagged either **Measured from audio** or **Inferred · NLP estimate**.

## Architecture

```
app.py                     Streamlit entry point: navigation, startup checks, offline badge
config.py                  Paths, defaults, speaker palette; disables telemetry before ML imports
views/                     One file per page (Dashboard, Upload, Speaker Analysis, Timeline,
                           Intelligence, Comparison, Reports, Settings)
ui/components.py           Shared cards, tags, chart styling (all user text HTML-escaped)
core/                      Pure processing modules (no UI)
  audio_validator.py         Extension allow-list, magic-byte sniffing, size limits
  audio_loader.py            FFmpeg/libsndfile decode to 16 kHz mono
  audio_preprocessor.py      DC removal, normalisation, RMS envelope, canonical WAV
  diarization_engine.py      pyannote pipeline (local cache only)
  transcription_engine.py    Faster-Whisper (local cache only)
  model_cache.py             Finds cached models without network calls
  speaker_alignment.py       Word/segment ↔ speaker alignment with confidence
  intervals.py               Union / intersection / subtraction / multi-coverage
  speaking_time.py, overlap_detection.py, interruption_analysis.py,
  silence_detection.py, participation_analysis.py,
  question_analysis.py, sentiment_analysis.py, conversation_summary.py, report_generator.py
services/
  analysis_service.py        Pipeline orchestration, background job runner, recompute after corrections
  model_manager.py           Model status, downloads, offline-readiness checks
  storage_service.py         Safe file storage, deletion, retention
  export_service.py          CSV / JSON exports
database/                  SQLAlchemy models, session handling, repository
scripts/download_models.py One-time model download / status check
tests/                     Pytest suite (ML mocked) + opt-in real-model integration test
data/                      Created at runtime: uploads, processed audio, reports, SQLite DB, logs
```

Pipeline: **decode → preprocess → transcribe → diarize → align → analytics → NLP → save**. Analysis runs in
a background thread, one job at a time to keep memory bounded. Models are cached in memory between runs (this
can be turned off in Settings).

> The page modules live in `views/` rather than `pages/`. A `pages/` folder triggers Streamlit's legacy
> auto-navigation, which overrode the app's sectioned sidebar.

## Technology stack

Streamlit · Plotly · custom CSS · Python 3.12 · pyannote.audio 4 · Faster-Whisper · PyTorch (CPU) · librosa ·
SoundFile · FFmpeg · spaCy · VADER · SQLite/SQLAlchemy · ReportLab · pandas · pytest

---

## Windows installation

### Prerequisites
- Windows 10/11, 8 GB RAM minimum (16 GB recommended), about 3 GB of free disk space
- **Python 3.11 or 3.12.** Python 3.13/3.14 may lack wheels for torch, pyannote or CTranslate2.
  [`uv`](https://docs.astral.sh/uv/) is optional but makes setup faster.
- **FFmpeg** for MP3/M4A/AAC/OGG/WEBM: `winget install Gyan.FFmpeg`, then open a new terminal.
  Without FFmpeg only WAV/FLAC can be decoded.

### Automatic setup
```bat
setup_windows.bat
```
This creates `.venv`, installs CPU-only PyTorch (avoiding multi-GB CUDA downloads), installs the requirements
and the spaCy English model, creates `.env` from `.env.example`, and downloads Whisper `small`.

### Manual setup
```bat
uv venv --python 3.12 .venv            REM or: py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
copy .env.example .env
python scripts\download_models.py --whisper small
```

## Hugging Face model access (speaker diarization)

The pyannote Community-1 model is free but *gated*: you accept its conditions once.

1. Create a free account at <https://huggingface.co/join>.
2. Open <https://huggingface.co/pyannote/speaker-diarization-community-1>, log in, and accept the conditions.
3. Create a **read** token at <https://huggingface.co/settings/tokens>.
4. Put it in `.env` (this file is git-ignored; never commit it):
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
   ```
5. Download the model:
   ```bat
   .venv\Scripts\python scripts\download_models.py --diarization
   ```
   You can also use **Settings → Models → Download diarization model**.

The token is needed only for this download. Afterwards the model loads from the local cache with no network
access, and you may remove the token. Until diarization is installed, the app still works in **single-speaker
mode**, which the UI labels clearly.

## Model download and sizes

```bat
.venv\Scripts\python scripts\download_models.py --check
.venv\Scripts\python scripts\download_models.py --whisper medium
```

| Whisper size | Download | RAM (int8) | Notes |
|---|---|---|---|
| tiny | 75 MB | ~0.5 GB | Fastest, lowest accuracy |
| base | 145 MB | ~0.7 GB | |
| **small** (default) | 480 MB | ~1.2 GB | Best balance on a 16 GB CPU laptop |
| medium | 1.5 GB | ~2.5 GB | More accurate, about 3× slower than small |

Models are stored in `%USERPROFILE%\.cache\huggingface\hub`. Only one Whisper model is kept in memory at a time.

## Running

```bat
run_windows.bat
```
Or run `.venv\Scripts\streamlit run app.py`, then open <http://localhost:8501>. The server binds to localhost only.

## Offline operation

Once the models are downloaded:
- The app starts and runs with no internet connection. The sidebar shows **Offline ready**, and
  **Settings → System status** checks the packages, models, FFmpeg, database and storage permissions.
- Models are loaded by local path (`local_files_only`). No Hub request is made at analysis time.
- Telemetry is disabled: Streamlit usage stats (`.streamlit/config.toml`), Hugging Face (`HF_HUB_DISABLE_TELEMETRY`)
  and pyannote metrics (`PYANNOTE_METRICS_ENABLED=0`).
- Ollama is optional, off by default, and only `localhost` URLs are accepted.
- Fonts are system fonts. No CDN or web-font requests are made.

To verify, disconnect from the network, start the app, and analyse a recording.

## Testing

```bat
.venv\Scripts\python -m pytest -q
```
The suite needs neither internet nor real models: pyannote and Whisper inference are mocked, and the analytics
are checked against synthetic segments with known answers. It covers validation, loading, interval math,
speaking time and percentages, overlap and interruption classification, silence, participation, alignment,
questions, sentiment, insight extraction, PDF/CSV generation, database CRUD and cascade deletion, retention,
offline and missing-model handling, the end-to-end pipeline, manual corrections, and a render test of every page.

**Real-model integration test (opt-in):**
```bat
set TALKTIME_INTEGRATION=1
set TALKTIME_TEST_AUDIO=C:\path\to\conversation.mp3
set TALKTIME_TEST_SPEAKERS=3
.venv\Scripts\python -m pytest tests\test_integration.py -s
```

## Data storage

| What | Where |
|---|---|
| Original uploads (optional; see Settings) | `data/uploads/` |
| Processed 16 kHz WAV used for playback | `data/processed/` |
| PDF reports | `data/reports/` |
| SQLite database | `data/database/talktime.db` |
| Logs (operational events only, never transcript text) | `data/logs/` |

Set `TALKTIME_DATA_DIR` in `.env` to store data elsewhere. Audio is never stored inside SQLite.

## Privacy and security

- All processing is local. Uploaded files are decoded by FFmpeg/libsndfile and **never executed**.
- Uploads are checked against an extension allow-list, their content signature (executables and documents are
  rejected), a size limit, and free disk space. Files are stored under random unique names with path-traversal
  protection.
- Deletion only removes files inside the app's data folders. Deleting a recording removes its database rows
  (cascade), audio files and reports.
- **Settings → Privacy & data** deletes a single recording or all analysis data. Optional automatic retention
  deletes recordings older than N days.
- No credentials are stored in code. `.env` is git-ignored.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Diarization model missing" | Follow *Hugging Face model access*. A 401/403 error means the conditions weren't accepted with the token's account. |
| "Could not decode audio" | Install FFmpeg, or convert the file to WAV. The file may be corrupted. |
| Analysis is slow | Use Whisper `base`, lower the beam size, or raise CPU threads in Settings. Measured on a Ryzen 7 5800H with a 50 s, 3-speaker MP3: about 58 s end to end (transcription ~12 s with Whisper `small`, diarization ~43 s). Diarization is the slowest stage on CPU. |
| High memory use | Settings → Processing → untick *Keep models in memory*, or use **Release loaded models**. |
| `pip` fails installing torch | Use Python 3.11/3.12 and the CPU index URL shown above. |
| Port already in use | `streamlit run app.py --server.port 8502` |

## Limitations

- Diarization can merge similar voices or split one voice into two. Use **Merge** and **Correct** to fix this.
  Labels are anonymous until renamed.
- Overlap is credited to every active speaker, so per-speaker shares of the recording don't sum to 100%. Each
  percentage states its denominator.
- "Possible interruption" describes timing, not intent. Unequal participation is not judged as good or bad.
- Question answering, topics, decisions and action items are heuristic and English-focused. Sentiment is
  text-only VADER (English) and does not measure emotion or mental state.
- CPU processing is not real-time. The actual processing time is measured and shown for every recording.
- Speakers are not identified across recordings. "Speaker 1" in two calls may be different people.
