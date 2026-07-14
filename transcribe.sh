#!/bin/bash
# transcribe.sh — Batch video/audio transcription using whisper.cpp
# Usage: ./transcribe.sh [options] [path ...]

set -Euo pipefail

LOCK_FILE="$HOME/.transcribe.lck"

# --- Single-instance lock (stale PID auto-clears) ---
acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local old_pid
        old_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            err "Another transcription is already running (PID $old_pid)"
            err "  Lock file: $LOCK_FILE"
            err "  Wait for it to finish or run: rm -f \"$LOCK_FILE\""
            exit 1
        fi
        rm -f "$LOCK_FILE"
    fi
    echo "$$" > "$LOCK_FILE"
}
release_lock() {
    [ -f "$LOCK_FILE" ] && [ "$(cat "$LOCK_FILE" 2>/dev/null)" = "$$" ] && rm -f "$LOCK_FILE"
}

# --- Early check: --background / -b (before getopt) ---
bg_mode=false
filtered_args=()
for a in "$@"; do
    case "$a" in
        -b|--background) bg_mode=true ;;
        *) filtered_args+=("$a") ;;
    esac
done

if [ "$bg_mode" = true ]; then
    LOG_FILE="$HOME/transcribe-$(date +%Y%m%d-%H%M%S).log"
    nohup bash "$0" "${filtered_args[@]}" > "$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "${LOG_FILE}.pid"
    echo "Background transcription started (PID: $pid)"
    echo "  Log:     $LOG_FILE"
    echo "  Tail:    tail -f \"$LOG_FILE\""
    exit 0
fi

# --- Defaults ---
MODEL="$HOME/Whisper Models/ggml-large-v3-turbo-q5_0.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
THREADS="$(nproc)"
LANGUAGE="auto"
PARALLEL=1
FORMATS="srt,txt"
OUTDIR=""
DRY_RUN=false
VERBOSE=false
QUIET=false
LOG_FILE=""
EXTENSIONS=(mp4 mkv mov avi webm m4v ts flac mp3 ogg wav)
RECURSIVE=true
MIN_SIZE=0
MAX_SIZE=0
SKIP_EXISTING=true
MANY_VIDS=false
KEEP_WAV=false
PRINT_PROGRESS=false
VAD=false
VAD_MODEL=""
NO_GPU=true
NO_FLASH_ATTN=false
TRANSLATE=false
FFMPEG_CONVERT=true
CURRENT_TMP_WAV=""
COOLDOWN=0
CANCEL_REQUESTED=false

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" >&2; }
err()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }
log()   { [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

cleanup() {
    local pids
    pids=$(jobs -rp 2>/dev/null || true)
    [ -n "$pids" ] && kill "$pids" 2>/dev/null || true
    [ -n "$CURRENT_TMP_WAV" ] && [ -f "$CURRENT_TMP_WAV" ] && rm -f "$CURRENT_TMP_WAV" || true
    release_lock
    return 0
}
trap cleanup EXIT TERM
handle_sigint() {
    if [ "$CANCEL_REQUESTED" = true ]; then
        echo "" >&2
        err "Force-cancelling..."
        exit 1
    fi
    CANCEL_REQUESTED=true
    echo "" >&2
    warn "Ctrl+C pressed — finishing current file, then stopping"
    warn "Press Ctrl+C again to force-cancel immediately"
    release_lock
}
trap handle_sigint INT

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [path ...]

Transcribe video/audio files using whisper.cpp (v1.9.1).

Options:
  -m, --model PATH       Model file or alias (tiny/base/small/medium/large-v3/turbo/turbo-q5, default: large-v3-turbo-q5)
  -t, --threads NUM      Thread count (default: $THREADS)
  -l, --lang LANG        Language code or "auto" (default: auto)
  -p, --parallel NUM     Parallel jobs (default: 1, each loads model independently)
  -f, --formats LIST     Output formats: srt,txt,vtt,json,tsv,lrc,csv (default: $FORMATS)
  -o, --outdir DIR       Output directory (default: same as source)
  -s, --min-size BYTES   Skip files smaller than this (default: 0)
  -S, --max-size BYTES   Skip files larger than this (default: 0 = no limit)
      --no-skip          Re-transcribe even if output exists
  -R, --no-recursive     Do not search subdirectories
  -n, --dry-run          Show what would be processed
      --many-vids        Show video count, total size, and suggested parallel mode
  -v, --verbose          Show whisper-cli command before execution
  -q, --quiet            Suppress per-file progress
      --vad              Enable voice activity detection
      --vad-model PATH   VAD model file (required with --vad)
      --translate        Translate to English
      --no-gpu           Disable GPU inference
      --no-flash-attn    Disable flash attention
      --no-convert       Skip ffmpeg conversion for video containers
      --keep-wav         Keep converted WAV file for debugging
      --print-progress   Show whisper.cpp progress bar (stderr)
      --cooldown SECS    Pause SECS seconds between files to avoid thermal throttling (default: 0)
  -L, --log FILE         Write log to file
  -h, --help             Show this help
  -b, --background       Run in background (auto-log to ~/transcribe-<date>.log)
EOF
}

# --- Parse args ---
TEMP=$(getopt -o bm:t:l:p:f:o:s:S:nqRvL:h \
    --long model:,threads:,lang:,parallel:,formats:,outdir:,min-size:,max-size:,no-skip,no-recursive,dry-run,quiet,verbose,many-vids,background,keep-wav,vad,vad-model:,translate,no-gpu,no-flash-attn,no-convert,print-progress,cooldown:,log:,help \
    -n "$(basename "$0")" -- "$@") || { usage; exit 1; }
eval set -- "$TEMP"

while true; do
    case "$1" in
        -m|--model)          MODEL="$2"; shift 2 ;;
        -t|--threads)        THREADS="$2"; shift 2 ;;
        -l|--lang)           LANGUAGE="$2"; shift 2 ;;
        -p|--parallel)       PARALLEL="$2"; shift 2 ;;
        -f|--formats)        FORMATS="$2"; shift 2 ;;
        -o|--outdir)         OUTDIR="$2"; shift 2 ;;
        -s|--min-size)       MIN_SIZE="$2"; shift 2 ;;
        -S|--max-size)       MAX_SIZE="$2"; shift 2 ;;
        --no-skip)           SKIP_EXISTING=false; shift ;;
        -b|--background)     shift ;;
        -R|--no-recursive)   RECURSIVE=false; shift ;;
        -n|--dry-run)        DRY_RUN=true; shift ;;
        --many-vids)         MANY_VIDS=true; shift ;;
        -q|--quiet)          QUIET=true; shift ;;
        -v|--verbose)        VERBOSE=true; shift ;;
        --vad)               VAD=true; shift ;;
        --vad-model)         VAD_MODEL="$2"; shift 2 ;;
        --translate)         TRANSLATE=true; shift ;;
        --no-gpu)            NO_GPU=true; shift ;;
        --no-flash-attn)     NO_FLASH_ATTN=true; shift ;;
        --no-convert)        FFMPEG_CONVERT=false; shift ;;
        --keep-wav)          KEEP_WAV=true; shift ;;
        --print-progress)    PRINT_PROGRESS=true; shift ;;
        --cooldown)          COOLDOWN="$2"; shift 2 ;;
        -L|--log)            LOG_FILE="$2"; shift 2 ;;
        -h|--help)           usage; exit 0 ;;
        --)                  shift; break ;;
        *)                   err "Unknown option: $1"; exit 1 ;;
    esac
done

# --print-progress overrides --quiet (can't pass both -np and -pp to whisper)
if [ "$PRINT_PROGRESS" = true ]; then
    QUIET=false
fi

# --- Resolve model aliases ---
resolve_model_alias() {
    case "${1,,}" in
        tiny)          echo "ggml-tiny.bin" ;;
        base)          echo "ggml-base.bin" ;;
        small)         echo "ggml-small.bin" ;;
        medium)        echo "ggml-medium.bin" ;;
        large-v1)      echo "ggml-large-v1.bin" ;;
        large-v2)      echo "ggml-large-v2.bin" ;;
        large-v3)      echo "ggml-large-v3.bin" ;;
        large-v3-turbo|turbo) echo "ggml-large-v3-turbo.bin" ;;
        turbo-q5)      echo "ggml-large-v3-turbo-q5_0.bin" ;;
        *)             echo "" ;;
    esac
}
alias_resolved=$(resolve_model_alias "$MODEL")
if [ -n "$alias_resolved" ]; then
    MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$alias_resolved"
    MODEL="$HOME/Whisper Models/$alias_resolved"
fi

TARGETS=("${@:-.}")

# --- Validate dependencies ---
WHISPER_BIN="${WHISPER_BIN:-whisper-cli}"

build_whisper_cli() {
    local tmp_dir deps_help
    deps_help=$(detect_distro_deps)
    command -v git &>/dev/null || { err "git not found. $deps_help"; return 1; }
    command -v cmake &>/dev/null || { err "cmake not found. $deps_help"; return 1; }
    if ! command -v gcc &>/dev/null || ! command -v g++ &>/dev/null; then
        err "C/C++ compiler not found. $deps_help"
        return 1
    fi

    tmp_dir=$(mktemp -d)
    info "Cloning whisper.cpp to $tmp_dir ..."
    if ! git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$tmp_dir"; then
        rm -rf "$tmp_dir"; return 1
    fi
    (
        cd "$tmp_dir" || exit 1
        info "Configuring build..."
        cmake -B build -DCMAKE_BUILD_TYPE=Release || exit 1
        info "Compiling (this may take a few minutes)..."
        cmake --build build -j || exit 1
        if [ -f build/bin/whisper-cli ]; then
            if sudo cp build/bin/whisper-cli /usr/bin/whisper-cli; then
                ok "Installed to /usr/bin/whisper-cli"
            else
                err "Install failed. Try: sudo cp $tmp_dir/build/bin/whisper-cli /usr/bin/"
                exit 1
            fi
        else
            err "Binary not found after build."
            ls build/bin/ 2>/dev/null
            exit 1
        fi
    )
    local rc
    rc=$?
    rm -rf "$tmp_dir"
    return "$rc"
}

detect_distro_deps() {
    if command -v apt &>/dev/null; then
        echo "Install with: sudo apt install git cmake build-essential"
    elif command -v dnf &>/dev/null; then
        echo "Install with: sudo dnf install git cmake gcc-c++ make"
    elif command -v pacman &>/dev/null; then
        echo "Install with: sudo pacman -S git cmake base-devel"
    elif command -v zypper &>/dev/null; then
        echo "Install with: sudo zypper install git cmake gcc-c++ make"
    elif command -v apk &>/dev/null; then
        echo "Install with: apk add git cmake gcc g++ make"
    else
        echo "Install: git, cmake, and a C/C++ compiler"
    fi
}

if ! command -v "$WHISPER_BIN" &>/dev/null; then
    warn "whisper-cli not found in PATH"
    echo -n "Build whisper.cpp from source? [Y/n]: "; read -r ans
    case "$ans" in
        n|N|no|No) err "Set WHISPER_BIN env var to a custom path."; exit 1 ;;
        *) build_whisper_cli || exit 1 ;;
    esac
fi

# --- Validate model ---
download_model() {
    local dir
    dir=$(dirname "$MODEL")
    mkdir -p "$dir"
    info "Downloading model to $MODEL ..."
    curl -L -o "$MODEL" "$MODEL_URL"
    ok "Model downloaded"
}

if [ "$MODEL" = "download" ]; then
    download_model
elif [ ! -f "$MODEL" ]; then
    fallback=$(find "$HOME/Whisper Models" -maxdepth 1 -name 'ggml-*.bin' 2>/dev/null | head -1)
    if [ -n "$fallback" ]; then
        warn "Model not found at $MODEL, using $fallback instead"
        MODEL="$fallback"
    else
        warn "No model found. Select a model to download:"
        echo ""
        echo "  1) tiny        (77 MB)    - fastest, least accurate"
        echo "  2) base        (148 MB)   - fast"
        echo "  3) small       (488 MB)   - good balance"
        echo "  4) medium      (1.5 GB)   - accurate"
        echo "  5) large-v3    (3.1 GB)   - most accurate"
        echo "  6) large-v3-turbo (1.6 GB) - fast + accurate"
        echo "  7) large-v3-turbo-q5_0 (574 MB) - recommended"
        echo "  8) tiny.en     (77 MB)    - English only, fastest"
        echo "  9) base.en     (148 MB)   - English only, fast"
        echo " 10) small.en    (488 MB)   - English only, good balance"
        echo " 11) medium.en   (1.5 GB)   - English only, accurate"
        echo " 12) Custom filename from ggerganov/whisper.cpp"
        echo ""
        echo -n "Choose [1-12] (default: 7): "; read -r choice
        case "$choice" in
            1|tiny)        model_file="ggml-tiny.bin" ;;
            2|base)        model_file="ggml-base.bin" ;;
            3|small)       model_file="ggml-small.bin" ;;
            4|medium)      model_file="ggml-medium.bin" ;;
            5|large-v3)    model_file="ggml-large-v3.bin" ;;
            6|large-v3-turbo) model_file="ggml-large-v3-turbo.bin" ;;
            7|turbo-q5|"") model_file="ggml-large-v3-turbo-q5_0.bin" ;;
            8|tiny.en)     model_file="ggml-tiny.en.bin" ;;
            9|base.en)     model_file="ggml-base.en.bin" ;;
            10|small.en)   model_file="ggml-small.en.bin" ;;
            11|medium.en)  model_file="ggml-medium.en.bin" ;;
            12|custom)     echo -n "Enter model filename (e.g. ggml-large-v3-q5_0.bin): "; read -r model_file; [ -z "$model_file" ] && model_file="ggml-large-v3-turbo-q5_0.bin" ;;
            q|Q|quit|exit) exit 1 ;;
            *)             model_file="ggml-large-v3-turbo-q5_0.bin" ;;
        esac
        MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$model_file"
        MODEL="$HOME/Whisper Models/$model_file"
        download_model
    fi
fi

# --- Validate VAD model ---
if [ "$VAD" = true ] && [ -z "$VAD_MODEL" ]; then
    err "--vad requires --vad-model PATH"
    exit 1
fi

# --- Single-instance lock ---
if [ "$DRY_RUN" = false ]; then
    acquire_lock
fi

# --- Find files ---
find_files() {
    local path="$1" resolved
    resolved="$(realpath -q "$path" 2>/dev/null || readlink -f "$path" 2>/dev/null || echo "$path")"
    if [ -f "$resolved" ]; then
        echo "$resolved"
    elif [ -d "$resolved" ]; then
        local cmd=(find "$resolved")
        [ "$RECURSIVE" = false ] && cmd+=(-maxdepth 1)
        cmd+=(-type f)
        local first=true
        local ext
        for ext in "${EXTENSIONS[@]}"; do
            if "$first"; then
                cmd+=( '(' -iname "*.$ext" )
                first=false
            else
                cmd+=( -o -iname "*.$ext" )
            fi
        done
        [ "$first" = false ] && cmd+=( ')' )
        "${cmd[@]}"
    fi
}

# --- Get output base path (without extension) for a given input ---
get_output_base() {
    local f="$1"
    if [ -n "$OUTDIR" ]; then
        local base_name
        base_name="$(basename "${f%.*}")"
        echo "$OUTDIR/$base_name"
    else
        echo "${f%.*}"
    fi
}

# --- Check if all requested output formats already exist and are fresh ---
output_exists() {
    local f="$1" out_base ext
    out_base=$(get_output_base "$f")
    ext="${f##*.}"
    local src_mtime
    src_mtime=$(stat -c%Y "$f" 2>/dev/null || echo 0)
    IFS=',' read -ra fmts <<< "$FORMATS"
    for fmt in "${fmts[@]}"; do
        local out_file=""
        case "$fmt" in
            srt)  out_file="$out_base.srt"; [ -f "$out_file" ] || out_file="$out_base.$ext.srt" ;;
            txt)  out_file="$out_base.txt"; [ -f "$out_file" ] || out_file="$out_base.$ext.txt" ;;
            vtt)  out_file="$out_base.vtt"; [ -f "$out_file" ] || out_file="$out_base.$ext.vtt" ;;
            json) out_file="$out_base.json"; [ -f "$out_file" ] || out_file="$out_base.$ext.json" ;;
            tsv)  out_file="$out_base.tsv"; [ -f "$out_file" ] || out_file="$out_base.$ext.tsv" ;;
            lrc)  out_file="$out_base.lrc"; [ -f "$out_file" ] || out_file="$out_base.$ext.lrc" ;;
            csv)  out_file="$out_base.csv"; [ -f "$out_file" ] || out_file="$out_base.$ext.csv" ;;
        esac
        [ -n "$out_file" ] || return 1
        [ -f "$out_file" ] || return 1
        [ -s "$out_file" ] || return 1
        local out_mtime
        out_mtime=$(stat -c%Y "$out_file" 2>/dev/null || echo 0)
        [ "$out_mtime" -ge "$src_mtime" ] || return 1
    done
    return 0
}

# --- Audio formats whisper-cli can read natively (via miniaudio) ---
AUDIO_EXTS=(flac mp3 ogg wav)
NEEDS_CONVERT() {
    local ext="$1"
    for a in "${AUDIO_EXTS[@]}"; do
        [ "${ext,,}" = "$a" ] && return 1
    done
    return 0
}

# --- Convert video to temp WAV via ffmpeg ---
conv_to_wav() {
    local src="$1" wav
    wav="$(mktemp /tmp/whisper_XXXXXX.wav)"
    ffmpeg -y -v quiet -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$wav"
    echo "$wav"
}

# --- Build and run whisper-cli ---
run_whisper() {
    local f="$1" orig_f="$1" ext tmp_wav=""
    ext="${f##*.}"
    if [ "$FFMPEG_CONVERT" = true ] && NEEDS_CONVERT "$ext"; then
        command -v ffmpeg &>/dev/null || { err "ffmpeg required for $ext files. Install ffmpeg or use --no-convert."; return 1; }
        tmp_wav=$(conv_to_wav "$f")
        CURRENT_TMP_WAV="$tmp_wav"
        f="$tmp_wav"
    fi
    local -a cmd=("$WHISPER_BIN" -t "$THREADS" -f "$f" -m "$MODEL")

    [ "$VERBOSE" = true ] && echo "       ${cmd[*]}" >&2

    [ "$LANGUAGE" != "auto" ] && cmd+=(-l "$LANGUAGE")
    [ "$TRANSLATE" = true ] && cmd+=(-tr)
    [ "$NO_GPU" = true ] && cmd+=(-ng)
    [ "$NO_FLASH_ATTN" = true ] && cmd+=(-nfa)
    [ "$QUIET" = true ] && cmd+=(-np)
    [ "$PRINT_PROGRESS" = true ] && cmd+=(-pp)

    if [ "$VAD" = true ]; then
        cmd+=(--vad)
        [ -n "$VAD_MODEL" ] && cmd+=(-vm "$VAD_MODEL")
    fi

    IFS=',' read -ra fmts <<< "$FORMATS"
    for fmt in "${fmts[@]}"; do
        case "$fmt" in
            srt)  cmd+=(-osrt)  ;;
            txt)  cmd+=(-otxt)  ;;
            vtt)  cmd+=(-ovtt)  ;;
            json) cmd+=(-oj)    ;;
            tsv)  cmd+=(-otsv)  ;;
            lrc)  cmd+=(-olrc)  ;;
            csv)  cmd+=(-ocsv)  ;;
            *)    warn "Unknown format: $fmt" ;;
        esac
    done

    local out_base
    if [ -n "$OUTDIR" ]; then
        out_base="$(basename "${orig_f%.*}")"
        out_base="$OUTDIR/$out_base"
    else
        out_base="${orig_f%.*}"
    fi
    cmd+=(-of "$out_base")

    local had_errexit=false
    shopt -qo errexit && had_errexit=true
    set +e
    "${cmd[@]}"
    local rc=$?
    $had_errexit && set -e
    [ -n "$tmp_wav" ] && [ "$KEEP_WAV" = false ] && rm -f "$tmp_wav"
    CURRENT_TMP_WAV=""
    return "$rc"
}

# Returns: 0=pass, 1=fail, 2=skip
transcribe_one() {
    local f="$1"

    # Size checks
    if [ "$MIN_SIZE" -gt 0 ] || [ "$MAX_SIZE" -gt 0 ]; then
        local fsize
        fsize=$(stat -c%s "$f" 2>/dev/null || echo 0)
        if [ "$MIN_SIZE" -gt 0 ] && [ "$fsize" -lt "$MIN_SIZE" ]; then
            [ "$QUIET" = false ] && echo -e "${YELLOW}SKIP${NC}  $f (${fsize}B < ${MIN_SIZE}B)"
            log "SKIP $f (too small: ${fsize}B)"
            return 2
        fi
        if [ "$MAX_SIZE" -gt 0 ] && [ "$fsize" -gt "$MAX_SIZE" ]; then
            [ "$QUIET" = false ] && echo -e "${YELLOW}SKIP${NC}  $f (${fsize}B > ${MAX_SIZE}B)"
            log "SKIP $f (too large: ${fsize}B)"
            return 2
        fi
    fi

    # Skip if output already exists
    if [ "$SKIP_EXISTING" = true ] && output_exists "$f"; then
        [ "$QUIET" = false ] && echo -e "${YELLOW}SKIP${NC}  $f (output exists)"
        log "SKIP $f (output exists)"
        return 2
    fi

    echo -e "${CYAN}TRANS${NC} $f"
    log "START $f"

    if [ "$DRY_RUN" = true ]; then
        echo "       $WHISPER_BIN -t $THREADS -f \"$f\" -m \"$MODEL\" ..."
        log "DRY   $f"
        return 0
    fi

    if run_whisper "$f"; then
        echo -e "${GREEN}DONE${NC}  $f"
        log "DONE  $f"
        return 0
    else
        local _ec=$?
        echo -e "${RED}FAIL${NC}  $f (exit $_ec)"
        log "FAIL  $f (exit $_ec)"
        return "$_ec"
    fi
}

# --- Collect files ---
all_files=()
for target in "${TARGETS[@]}"; do
    while IFS= read -r f; do
        all_files+=("$f")
    done < <(find_files "$target")
done

# Sort by file size (smallest first) — quicker files transcribed first
if [ "${#all_files[@]}" -gt 1 ]; then
    sorted=()
    while IFS= read -r f; do
        sorted+=("$f")
    done < <(for f in "${all_files[@]}"; do printf '%020d\t%s\n' "$(stat -c%s "$f" 2>/dev/null)" "$f"; done | sort -n | cut -f2-)
    all_files=("${sorted[@]}")
fi

total="${#all_files[@]}"
if [ "$total" -eq 0 ]; then
    warn "No supported files found in: ${TARGETS[*]}"
    exit 0
fi

info "Found $total file(s) to process"
[ -n "$LOG_FILE" ] && echo -e "  Log:     ${CYAN}$LOG_FILE${NC}"


if [ "$MANY_VIDS" = true ]; then
    total_bytes=0
    for f in "${all_files[@]}"; do
        sizes=$(stat -c%s "$f" 2>/dev/null || echo 0)
        total_bytes=$((total_bytes + sizes))
    done
    if [ "$total_bytes" -gt 1073741824 ]; then
        fmt_size=$(awk "BEGIN{printf \"%.1f GB\", $total_bytes/1073741824}")
    elif [ "$total_bytes" -gt 1048576 ]; then
        fmt_size=$(awk "BEGIN{printf \"%.1f MB\", $total_bytes/1048576}")
    else
        fmt_size=$(awk "BEGIN{printf \"%.0f KB\", $total_bytes/1024}")
    fi
    suggested=$(( $(nproc) / 2 ))
    [ "$suggested" -lt 1 ] && suggested=1
    [ "$suggested" -gt 4 ] && suggested=4
    echo "  Total size:  $fmt_size"
    echo "  Formats:     ${EXTENSIONS[*]}"
    echo "  Output:      ${FORMATS//,/, }"
    echo "  Suggested:   transcribe.sh -p $suggested [path ...]"
    echo ""
    exit 0
fi

[ "$DRY_RUN" = true ] && info "=== DRY RUN - no files will be transcribed ==="
echo ""

# --- Process ---
count_passed=0
count_failed=0
count_skipped=0
failed_files=()

if [ "$PARALLEL" -gt 1 ]; then
    [ "$CANCEL_REQUESTED" = true ] && { warn "Cancelled by user"; exit 1; }
    info "Processing $total file(s) with $PARALLEL parallel job(s)..."

    temp_dir=$(mktemp -d)
    trap 'cleanup; rm -rf "$temp_dir"' EXIT
    export TEMP_DIR="$temp_dir"
    export WHISPER_BIN FFMPEG_CONVERT OUTDIR MODEL LANGUAGE THREADS FORMATS MIN_SIZE MAX_SIZE CURRENT_TMP_WAV KEEP_WAV COOLDOWN
    export SKIP_EXISTING QUIET VERBOSE LOG_FILE VAD VAD_MODEL TRANSLATE NO_GPU NO_FLASH_ATTN PRINT_PROGRESS
    parallel_worker() {
        local f="$1" rc h
        trap '[ -n "$CURRENT_TMP_WAV" ] && rm -f "$CURRENT_TMP_WAV"' EXIT INT TERM
        transcribe_one "$f"
        rc=$?
        h=$(printf "%s" "$f" | md5sum | cut -c1-12)
        echo "$rc" > "$TEMP_DIR/result_$h"
        [ "$rc" -gt 0 ] && [ "$rc" -ne 2 ] && echo "$f" > "$TEMP_DIR/fname_$h"
    }
    export -f transcribe_one run_whisper output_exists get_output_base conv_to_wav NEEDS_CONVERT log info ok warn err parallel_worker
    export RED GREEN YELLOW CYAN NC

    mon_pid=""
    if [ "$QUIET" = false ] && [ "$total" -gt 0 ]; then
        (
            while true; do
                done_count=$(find "$temp_dir" -maxdepth 1 -name 'result_*' 2>/dev/null | wc -l)
                pct=$((done_count * 100 / total))
                printf "\r${CYAN}  → ${done_count}/${total} files processed (${pct}%%)${NC}" >&2
                [ "$done_count" -ge "$total" ] && break
                sleep 2
            done
            printf "\r${CYAN}  → ${total}/${total} files processed (100%%)${NC}" >&2
            echo "" >&2
        ) &
        mon_pid=$!
    fi

    printf '%s\0' "${all_files[@]}" | xargs -0 -P "$PARALLEL" -I{} bash -c "parallel_worker \"\$1\"" _ {} 2>&1

    [ -n "$mon_pid" ] && kill "$mon_pid" 2>/dev/null || true
    for rf in "$temp_dir"/result_*; do
        [ -f "$rf" ] || continue
        read -r code < "$rf"
        case "$code" in
            0) count_passed=$((count_passed + 1)) ;;
            2) count_skipped=$((count_skipped + 1)) ;;
            *) count_failed=$((count_failed + 1)); failed_files+=("$(cat "$TEMP_DIR/fname_$(basename "$rf" | sed 's/^result_//')" 2>/dev/null)") ;;
        esac
    done
else
    processed=0
    for f in "${all_files[@]}"; do
        [ "$CANCEL_REQUESTED" = true ] && break
        code=0
        set +e
        transcribe_one "$f"
        code=$?
        set -e
        processed=$((processed + 1))
        pct=$((processed * 100 / total))
        echo -e "${CYAN}  → ${processed}/${total} files processed (${pct}%)${NC}"
        case "$code" in
            0) count_passed=$((count_passed + 1)) ;;
            2) count_skipped=$((count_skipped + 1)) ;;
            *) count_failed=$((count_failed + 1)); failed_files+=("$f") ;;
        esac
        if [ "$COOLDOWN" -gt 0 ] && [ "$processed" -lt "$total" ]; then
            info "Cooling down for ${COOLDOWN}s..."
            set +e; sleep "$COOLDOWN"; set -e
        fi
    done
fi

# --- Summary ---
echo ""
info "=== Summary ==="
echo -e "  Total:   ${CYAN}$total${NC}"
echo -e "  Done:    ${GREEN}$count_passed${NC}"
echo -e "  Skipped: ${YELLOW}$count_skipped${NC}"
echo -e "  Failed:  ${RED}$count_failed${NC}"
[ "${#failed_files[@]}" -gt 0 ] && for ff in "${failed_files[@]}"; do echo "    - $ff"; done
[ -n "$LOG_FILE" ] && echo -e "  Log:     ${CYAN}$LOG_FILE${NC}"
echo ""

if command -v notify-send &>/dev/null; then
    notify-send "Transcription Complete" "Done: $count_passed  Skipped: $count_skipped  Failed: $count_failed" 2>/dev/null || true
fi
