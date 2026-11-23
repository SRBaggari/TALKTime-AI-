"""One-time model download (the only step that needs internet).

Usage (from the project folder):
    .venv\\Scripts\\python scripts\\download_models.py --whisper small
    .venv\\Scripts\\python scripts\\download_models.py --diarization
    .venv\\Scripts\\python scripts\\download_models.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from core.diarization_engine import SETUP_INSTRUCTIONS, is_diarization_available  # noqa: E402
from core.transcription_engine import is_whisper_available  # noqa: E402
from services import model_manager  # noqa: E402


def check() -> int:
    print("Local model status")
    for size in config.WHISPER_SIZES:
        print(f"  faster-whisper-{size:<7} {'cached' if is_whisper_available(size) else 'not downloaded'}")
    print(f"  pyannote Community-1   {'cached' if is_diarization_available() else 'not downloaded'}")
    print(f"  HF token configured    {'yes' if config.get_hf_token() else 'no'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--whisper", choices=config.WHISPER_SIZES, help="Download a Faster-Whisper model size")
    p.add_argument("--diarization", action="store_true", help="Download pyannote speaker-diarization-community-1")
    p.add_argument("--check", action="store_true", help="Only report what is cached")
    args = p.parse_args()
    if args.check or not (args.whisper or args.diarization):
        return check()
    if args.whisper:
        if is_whisper_available(args.whisper):
            print(f"faster-whisper-{args.whisper} is already cached.")
        else:
            print(f"Downloading faster-whisper-{args.whisper} …")
            print("Saved to", model_manager.download_whisper(args.whisper))
    if args.diarization:
        if is_diarization_available():
            print("pyannote Community-1 is already cached.")
        elif not config.get_hf_token():
            print("HF_TOKEN is not set.\n\n" + SETUP_INSTRUCTIONS)
            return 1
        else:
            print("Downloading pyannote/speaker-diarization-community-1 …")
            try:
                print("Saved to", model_manager.download_diarization())
            except Exception as exc:  # gated-repo errors are the common case
                print(f"Download failed: {exc.__class__.__name__}: {exc}")
                print("Make sure you accepted the model conditions on Hugging Face with the account that owns the token.")
                return 1
    print("\nDone. These models now work offline.")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
