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
| Transcription | 99 languages (English, Telugu, Hindi, Tamil listed first), auto-detection with probabilities, manual selection, word timestamps, `[unclear]` flags for low-confidence or repetitive output; never translated | Faster-Whisper (CTranslate2, CPU int8) |
| Code-switching | Each speech region gets its own language detection (e.g. Telugu ↔ English turns), per-line language tags, languages-spoken breakdown | VAD regions + Whisper language ID |
| Native script | Telugu/Tamil/Kannada/… that Whisper writes in Devanagari is converted back to the native script (original kept) | Deterministic Unicode transliteration |
| Audio quality | Score, speech level, noise floor, SNR, clipping, bandwidth, sample rate/channels, issues + recommendations | Measured on un-normalised audio |
| Voice activity detection | Speech vs non-speech on the original timeline; quiet vs background-sound gaps; VAD↔diarization agreement; skips models when there is no speech | Silero VAD (bundled, local) with energy fallback |
| Speaker-wise transcript | Each line assigned by strongest timestamp overlap, at word level; per-line and per-speaker assignment confidence; uncertain lines flagged | Interval alignment (pyannote gives no per-segment probability, so none is invented) |
| Speaking time | Total, share of speech, share of recording, turns, average/longest/shortest turn, first/last spoke | Exact interval math |
| Overlaps | Overlap events, *possible* interruptions, backchannels, simultaneous starts | Timing heuristics, conservative wording |
| Silence | Total/longest/average silence, gaps between speakers, pauses within a speaker, response delays | Speech-activity gaps + acoustic energy check |
| Participation | Balance (normalised entropy), Gini, dominance ratio, turns/min, response times, longest turn, dominance indicators | Documented formulas (shown in the app) |
| Questions | Questions per speaker, likely responses, possibly unanswered questions, confidence | Heuristics |
| Sentiment | Positive / neutral / negative per segment and speaker, trend over time (English) | VADER lexicon (text only) |
| Conversation intelligence | Topics, repeated topics (discussion episodes), key points, decisions, action items, commitments, important statements | spaCy + rule-based extraction (English); keyword topics for other languages |
| Transcript search | Keywords, exact phrases, speaker name, timestamps (`1:23`) across all recordings; highlighted results; click to play | SQLite + Unicode matching |
| Optional LLM summary | Abstractive summary | Your local Ollama (disabled by default, localhost only) |
| Timeline | Speaker lanes, overlap and silence lanes, click to play, play an individual segment | Plotly + synced audio |
| Corrections | Rename speakers, merge split speakers, reassign a transcript line; analytics recalculated | No re-transcription needed |
| Comparison | Duration, ranked speaking shares, balance, Gini, turn length, turns/min, interruptions/min, silence, pauses, response delay, sentiment, quality; historical trends | Speakers never matched across calls (no voiceprints stored) |
| Reports | PDF (charts, quality, language, dynamics, processing, limitations); Excel workbook (11+ sheets incl. one per speaker); CSV; JSON; speaker-wise TXT; interactive offline HTML charts | ReportLab, openpyxl, Plotly |
| Processing monitor | Live CPU/memory, real-time factor, time per stage, model load times, peak memory, failures with stage and recovery hint | psutil |
| Privacy | Local-only indicator, typed confirmation for every deletion; delete recording / transcript+analysis / reports / original upload / everything; retention; masked token with one-click removal | — |
| History | All analyses stored in SQLite; delete one recording or everything | SQLAlchemy |

Every on-screen section is tagged either **Measured from audio** or **Inferred · NLP estimate**.

## Architecture

```
app.py                     Streamlit entry point: navigation, startup checks, offline badge
config.py                  Paths, defaults, speaker palette; disables telemetry before ML imports
views/                     One file per page (Dashboard, Upload, Speaker Analysis, Timeline, Intelligence,
                           Audio & Language, Transcript Search, Comparison, Reports, Processing Monitor, Settings)
ui/components.py           Shared cards, tags, chart styling (all user text HTML-escaped)
core/                      Pure processing modules (no UI)
  audio_validator.py         Extension allow-list, magic-byte sniffing, size limits
  audio_loader.py            FFmpeg/libsndfile decode to 16 kHz mono
  audio_preprocessor.py      DC removal, normalisation, RMS envelope, canonical WAV
  diarization_engine.py      pyannote pipeline (local cache only)
  transcription_engine.py    Faster-Whisper (local cache only)
  model_cache.py             Finds cached models without network calls
  speaker_alignment.py       Word/segment ↔ speaker alignment with confidence
  vad.py                     Voice activity detection, quiet vs background-sound classification
  audio_quality.py           Levels, SNR, clipping, bandwidth, issues and recommendations
  language_utils.py          Whisper languages, script detection, code-switch report, feature support
  transliteration.py         Devanagari → native Indic script
  transcript_search.py       Query parsing, timestamp search, safe highlighting
  comparison.py              Cross-recording summaries and trends
  intervals.py               Union / intersection / subtraction / multi-coverage
  speaking_time.py, overlap_detection.py, interruption_analysis.py,
  silence_detection.py, participation_analysis.py,
  question_analysis.py, sentiment_analysis.py, conversation_summary.py, report_generator.py
services/
  analysis_service.py        Pipeline orchestration, background job runner, recompute after corrections
  model_manager.py           Model status, downloads, offline-readiness checks
  storage_service.py         Safe file storage, deletion, retention
  export_service.py          CSV, JSON, XLSX, TXT and interactive HTML exports
  resource_monitor.py        CPU / memory sampling during analysis
database/                  SQLAlchemy models, session handling, repository
scripts/download_models.py One-time model download / status check
tests/                     Pytest suite (ML mocked) + opt-in real-model integration test
data/                      Created at runtime: uploads, processed audio, reports, SQLite DB, logs
```

Pipeline: **probe → decode → VAD → quality → transcribe → diarize → align → analytics → NLP → save**. Analysis runs in
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

## Multilingual recordings

- Pick a language on the upload page, or leave **Auto-detect**. The Audio & Language page shows the detected
  language with Whisper's probabilities.
- For conversations that switch language (e.g. Telugu ↔ English), turn on **Mixed languages / code-switching**.
  Each speech region is then transcribed with its own language detection. This is slower, but short turns in a
  second language are no longer lost.
- Speech is always transcribed in the language spoken. Nothing is translated.
- **Use Whisper `medium` for Telugu, Tamil, Hindi and other Indic languages.** In tests on synthetic Telugu speech,
  `small` produced partly garbled text while `medium` was close to the spoken words. The app shows this advice
  when relevant.
- Whisper often writes Telugu/Tamil/Kannada in Devanagari. TalkTime AI converts such lines back to the native
  script and keeps the original. Hover the "script converted" tag on the timeline to see it.
- Sentiment, spaCy topics and the decision/action-item rules are English-only. For other languages they are
  switched off rather than giving unreliable results. Transcription, all audio and speaker analytics, keyword
  topics and `?`-based questions still work.

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
| Analysis is slow | Use Whisper `base`, lower the beam size, or raise CPU threads in Settings. Measured on a Ryzen 7 5800H with a 50 s, 3-speaker English MP3: about 58 s end to end (transcription ~12 s with `small`, diarization ~43 s on the first run including model load). Code-switching mode and `medium` are slower. A 33 s Telugu/English clip took ~77 s with `small` and ~250 s with `medium`. See the Processing Monitor for your own measurements. |
| Telugu/Tamil transcript looks wrong | Switch to Whisper `medium` (Settings → Processing), or set the language manually instead of Auto-detect. Lines marked `[unclear]` were flagged as low-confidence or repetitive. |
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
- Speakers are not identified across recordings. "Speaker 1" in two calls may be different people. Voice
  matching across recordings is deliberately not offered, because it would require storing voiceprints (biometric data).
- Speaker-assignment confidence is alignment-based (how much diarized speech under a line belongs to the assigned
  speaker). The diarization model provides no per-segment probability, so none is shown.
- Code-switching works between speech regions (separate turns or sentences). Language switches *within* one
  sentence share that region's language label. The script-mix table helps spot these.
- Devanagari → Indic transliteration is a script conversion. It does not fix words Whisper misheard. Languages that
  share a script (Hindi/Marathi, English/Spanish) cannot be told apart by script.
- The SNR is an estimate (speech regions also contain noise). Gaps of pure digital silence are clamped at −90 dBFS.
