"""One-off: fetch YouTube transcript for bZsSxiUaw8w."""
from __future__ import annotations

import importlib.util
import sys

VIDEO_ID = "bZsSxiUaw8w"


def main() -> int:
    has_yta = importlib.util.find_spec("youtube_transcript_api") is not None
    has_ytdlp = importlib.util.find_spec("yt_dlp") is not None
    print(f"youtube_transcript_api={has_yta} yt_dlp={has_ytdlp}")

    if has_yta:
        from youtube_transcript_api import YouTubeTranscriptApi

        try:
            api = YouTubeTranscriptApi()
            fetched = api.fetch(VIDEO_ID)
            lines = [sn.text for sn in fetched]
            text = "\n".join(lines)
            out = f"transcripts/{VIDEO_ID}.txt"
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"saved {out} chars={len(text)}")
            print(text[:2000])
            return 0
        except Exception as e:
            print(f"youtube_transcript_api failed: {type(e).__name__}: {e}")

        try:
            # older API style
            transcript_list = YouTubeTranscriptApi.list_transcripts(VIDEO_ID)
            print("available:", [t.language_code for t in transcript_list])
            t = transcript_list.find_transcript(["en", "en-US", "a.en"])
            data = t.fetch()
            text = "\n".join(x["text"] for x in data)
            out = f"transcripts/{VIDEO_ID}.txt"
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"saved {out} chars={len(text)}")
            print(text[:2000])
            return 0
        except Exception as e:
            print(f"list_transcripts failed: {type(e).__name__}: {e}")

    if has_ytdlp:
        import yt_dlp

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US"],
            "subtitlesformat": "vtt",
            "outtmpl": f"transcripts/{VIDEO_ID}.%(ext)s",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={VIDEO_ID}", download=True
            )
            subs = info.get("subtitles") or {}
            autos = info.get("automatic_captions") or {}
            print("manual_subs", list(subs.keys())[:20])
            print("auto_subs", list(autos.keys())[:20])
        return 0

    print("No transcript libraries available")
    return 1


if __name__ == "__main__":
    sys.exit(main())
