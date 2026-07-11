# transcribe-video

A bash script that automatically converts video/audio files into text (transcription) using **whisper.cpp**.

Takes any video (MP4, MKV, AVI, etc.) or audio file (MP3, WAV, etc.) and generates subtitle files (.srt) and plain text files (.txt) with everything that was spoken.

---

## What does this script do?

This script finds all video and audio files in a folder (including subfolders), feeds them one-by-one to **whisper.cpp** (an AI model that listens to audio and writes down what people say), and saves the transcribed text as files next to the original video.

For example, if you have a file called `lecture.mp4`, the script will create:
- `lecture.srt` — subtitle file with timestamps (usable in any video player)
- `lecture.txt` — plain text of everything spoken

---

## Requirements (what you need installed)

### 1. whisper.cpp (`whisper-cli`)
This is the actual AI engine that does the transcribing.

**Install from source:**
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make -j
sudo make install
```

### 2. ffmpeg
Used to extract audio from video files (whisper.cpp only reads audio formats natively).

```bash
sudo apt install ffmpeg       # Ubuntu/Debian
brew install ffmpeg           # macOS
```

### 3. Model file (the AI brain)
A pre-trained model that whisper.cpp uses to recognize speech.

The script looks for the model at `~/models/ggml-large-v3-turbo-q5_0.bin` by default.

If the model is missing, the script will ask if you want to download it automatically.

**Manual download:**
```bash
curl -L -o ~/models/ggml-large-v3-turbo-q5_0.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin
```

---

## Quick Start

```bash
# Transcribe everything in the current folder (and subfolders)
./transcribe.sh

# Transcribe everything in a specific folder
./transcribe.sh "/path/to/your/videos/"

# Transcribe a single file
./transcribe.sh "myvideo.mp4"
```

That's it. The script will find all supported files, transcribe them one by one, and save `.srt` and `.txt` files beside each original file.

---

## Options (for advanced usage)

| Option | What it does |
|--------|-------------|
| `-m, --model PATH` | Use a different model file (default: `~/models/ggml-large-v3-turbo-q5_0.bin`) |
| `-t, --threads NUM` | How many CPU threads to use (default: all available cores) |
| `-l, --lang LANG` | Language code like `en`, `hi`, `es` or `auto` to detect (default: auto) |
| `-p, --parallel NUM` | Transcribe multiple files at once (default: 1) |
| `-f, --formats LIST` | Output formats: `srt,txt,vtt,json,tsv,lrc,csv` (default: `srt,txt`) |
| `-o, --outdir DIR` | Save output files to a specific folder instead of next to the video |
| `-s, --min-size BYTES` | Skip files smaller than this size |
| `-S, --max-size BYTES` | Skip files larger than this size |
| `--no-skip` | Re-transcribe even if output already exists |
| `-R, --no-recursive` | Don't search subfolders |
| `-n, --dry-run` | Show what would be processed without actually doing it |
| `-q, --quiet` | Less output on screen |
| `--vad` | Voice Activity Detection (skip silent parts) |
| `--vad-model PATH` | VAD model file (required with `--vad`) |
| `--translate` | Translate speech to English |
| `--no-gpu` | Disable GPU (use CPU only) |
| `--no-flash-attn` | Disable flash attention |
| `--no-convert` | Skip ffmpeg conversion (only useful for native audio files) |
| `--print-progress` | Show whisper.cpp's detailed progress |
| `-L, --log FILE` | Write a log of what was processed |
| `-h, --help` | Show all options |

### Examples

```bash
# Transcribe all Hindi videos in a folder, translate to English
./transcribe.sh -l hi --translate "/path/to/hindi/videos/"

# Process 4 files at once, output only JSON format
./transcribe.sh -p 4 -f json "./videos/"

# Dry run to check what files would be processed
./transcribe.sh -n "./my videos/"

# Save all transcripts to a separate folder
./transcribe.sh -o "./transcripts/" "./videos/"

# Skip files larger than 500MB
./transcribe.sh -S 524288000 "./videos/"
```

---

## How it works (step by step)

1. **Find files** — Scans the given folder (and all subfolders) for video/audio files
2. **Sort by size** — Smallest files processed first (so you get results faster)
3. **Convert to WAV** — Video files are converted to raw audio using ffmpeg (16kHz, mono)
4. **Run whisper.cpp** — Feeds the audio to the AI model and generates text
5. **Save output** — Writes `.srt`, `.txt` (or whatever formats you chose) next to the original file
6. **Clean up** — Deletes the temporary WAV file (even if you press Ctrl+C)

---

## Supported file types

**Video:** mp4, mkv, mov, avi, webm, m4v, ts  
**Audio:** flac, mp3, ogg, wav

---

## Troubleshooting

### `whisper-cli: error while loading shared libraries: libwhisper.so.1`

whisper.cpp is installed but the system can't find its library. Fix:

```bash
echo /usr/local/lib | sudo tee /etc/ld.so.conf.d/whisper.conf
sudo ldconfig
```

### `whisper-cli: command not found`

whisper.cpp is not installed. See the [Requirements](#1-whispercpp-whisper-cli) section above.

### `ffmpeg: command not found`

ffmpeg is not installed. Run `sudo apt install ffmpeg` or equivalent.

---

## Reference

- [whisper.cpp GitHub](https://github.com/ggerganov/whisper.cpp) — The C++ inference engine
- [whisper.cpp Models on Hugging Face](https://huggingface.co/ggerganov/whisper.cpp) — Pre-trained model downloads
- [OpenAI Whisper](https://github.com/openai/whisper) — The original research model
