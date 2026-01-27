import os
import re
import sys
from pathlib import Path

def extract_video_id(url_or_id: str) -> str:
    """
    Accepts:
      - full YouTube URL
      - youtu.be short URL
      - raw video id
    Returns: video_id
    """
    s = url_or_id.strip()

    # If it's already an 11-char-ish ID, just return it (basic check)
    if re.fullmatch(r"[A-Za-z0-9_-]{8,20}", s) and "http" not in s:
        return s

    # Pull v= from URL
    m = re.search(r"[?&]v=([A-Za-z0-9_-]{8,20})", s)
    if m:
        return m.group(1)

    # Pull from youtu.be/
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{8,20})", s)
    if m:
        return m.group(1)

    raise ValueError("Could not extract a YouTube video id from the input.")


def try_youtube_captions(video_id: str, lang_prefs=("en", "en-US", "en-GB")) -> str | None:
    """
    Works with newer AND older youtube-transcript-api versions.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        print("youtube-transcript-api not installed. Skipping caption fetch.")
        return None

    # Newer versions support list_transcripts; older ones don't.
    if hasattr(YouTubeTranscriptApi, "list_transcripts"):
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for lang in lang_prefs:
                try:
                    t = transcript_list.find_manually_created_transcript([lang]).fetch()
                    return "\n".join([x["text"] for x in t])
                except Exception:
                    pass

            for lang in lang_prefs:
                try:
                    t = transcript_list.find_generated_transcript([lang]).fetch()
                    return "\n".join([x["text"] for x in t])
                except Exception:
                    pass

            # fallback: first available transcript
            t = next(iter(transcript_list)).fetch()
            return "\n".join([x["text"] for x in t])

        except Exception as e:
            print(f"Caption fetch failed: {e}")
            return None

    # Older versions: use get_transcript directly
    try:
        t = YouTubeTranscriptApi.get_transcript(video_id, languages=list(lang_prefs))
        return "\n".join([x["text"] for x in t])
    except Exception as e:
        print(f"Caption fetch failed: {e}")
        return None



def download_audio(url: str, out_dir: Path) -> Path:
    """
    Downloads best audio WITHOUT conversion (no ffmpeg needed).
    Returns the downloaded audio file path (often .webm or .m4a).
    """
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "%(id)s.%(ext)s")

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "-o", output_template,
        url
    ]

    print("Downloading audio with yt-dlp (no ffmpeg conversion)...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("yt-dlp failed to download audio.")

    video_id = extract_video_id(url)

    # Find the downloaded file (any extension)
    candidates = [p for p in out_dir.glob(f"{video_id}.*") if not p.name.endswith(".part")]
    if not candidates:
        raise FileNotFoundError("Audio file not found after download.")

    # Pick largest file
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def transcribe_mp3(mp3_path: Path, model_size="small", language="en") -> str:
    """
    Offline transcription using faster-whisper.
    model_size options: tiny, base, small, medium, large-v2, large-v3
    """
    from faster_whisper import WhisperModel

    # Uses CPU by default. If you have CUDA, you can set device="cuda".
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print(f"Transcribing {mp3_path.name} with faster-whisper (model={model_size})...")
    segments, info = model.transcribe(str(mp3_path), language=language, vad_filter=True)

    lines = []
    for seg in segments:
        lines.append(seg.text.strip())
    return "\n".join(lines)


def main():
    # ===== HARD-CODED DEFAULTS =====
    DEFAULT_URL_OR_ID = "https://www.youtube.com/watch?v=joBmbh0AGSQ"
    DEFAULT_OUTPUT_FILE = "transcript.txt"

    # ===== ARG OVERRIDE (OPTIONAL) =====
    url_or_id = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_URL_OR_ID
    output_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path(DEFAULT_OUTPUT_FILE)

    print(f"Using video: {url_or_id}")
    print(f"Writing output to: {output_file.resolve()}")

    video_id = extract_video_id(url_or_id)
    url = url_or_id if "http" in url_or_id else f"https://www.youtube.com/watch?v={video_id}"

    # 1) Try captions
    text = try_youtube_captions(video_id)
    if text:
        output_file.write_text(text, encoding="utf-8")
        print(f"Saved transcript from captions to: {output_file.resolve()}")
        return

    # 2) Fallback: download audio → transcribe
    out_dir = Path("audio_tmp")
    audio_path = download_audio(url, out_dir=out_dir)

    text = transcribe_mp3(audio_path, model_size="small", language="en")
    output_file.write_text(text, encoding="utf-8")
    print(f"Saved transcript from audio transcription to: {output_file.resolve()}")



if __name__ == "__main__":
    main()
