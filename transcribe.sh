#!/bin/bash
# transcribe.sh — Batch video/audio transcription using whisper.cpp
# Usage: ./transcribe.sh [options] [path ...]

set -Euo pipefail

# --- Defaults ---
MODEL="$HOME/models/ggml-large-v3-turbo-q5_0.bin"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin"
THREADS="$(nproc)"
LANGUAGE="auto"
PARALLEL=1
FORMATS="srt,txt"
OUTDIR=""
DRY_RUN=false
QUIET=false
LOG_FILE=""
EXTENSIONS=(mp4 mkv mov avi webm m4v ts flac mp3 ogg wav)
RECURSIVE=true
MIN_SIZE=0
MAX_SIZE=0
SKIP_EXISTING=true
PRINT_PROGRESS=false
VAD=false
VAD_MODEL=""
NO_GPU=true
NO_FLASH_ATTN=false
TRANSLATE=false
FFMPEG_CONVERT=true

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
    [ -n "$pids" ] && kill $pids 2>/dev/null || true
}
trap cleanup EXIT

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [path ...]

Transcribe video/audio files using whisper.cpp (v1.9.1).

Options:
  -m, --model PATH       Model file path (default: $MODEL)
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
  -q, --quiet            Suppress per-file progress
      --vad              Enable voice activity detection
      --vad-model PATH   VAD model file (required with --vad)
      --translate        Translate to English
      --no-gpu           Disable GPU inference
      --no-flash-attn    Disable flash attention
      --no-convert       Skip ffmpeg conversion for video containers (will fail if whisper-cli can't read input)
      --print-progress   Show whisper.cpp progress bar (stderr)
  -L, --log FILE         Write log to file
  -h, --help             Show this help
EOF
    exit 0
}

# --- Parse args ---
TEMP=$(getopt -o m:t:l:p:f:o:s:S:nqRL:h \
    --long model:,threads:,lang:,parallel:,formats:,outdir:,min-size:,max-size:,no-skip,no-recursive,dry-run,quiet,vad,vad-model:,translate,no-gpu,no-flash-attn,no-convert,print-progress,log:,help \
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
        -R|--no-recursive)   RECURSIVE=false; shift ;;
        -n|--dry-run)        DRY_RUN=true; shift ;;
        -q|--quiet)          QUIET=true; shift ;;
        --vad)               VAD=true; shift ;;
        --vad-model)         VAD_MODEL="$2"; shift 2 ;;
        --translate)         TRANSLATE=true; shift ;;
        --no-gpu)            NO_GPU=true; shift ;;
        --no-flash-attn)     NO_FLASH_ATTN=true; shift ;;
        --no-convert)        FFMPEG_CONVERT=false; shift ;;
        --print-progress)    PRINT_PROGRESS=true; shift ;;
        -L|--log)            LOG_FILE="$2"; shift 2 ;;
        -h|--help)           usage ;;
        --)                  shift; break ;;
        *)                   err "Unknown option: $1"; exit 1 ;;
    esac
done

TARGETS=("${@:-.}")

# --- Validate dependencies ---
WHISPER_BIN="${WHISPER_BIN:-whisper-cli}"
command -v "$WHISPER_BIN" &>/dev/null || { err "whisper-cli not found. Install whisper.cpp or set WHISPER_BIN env var."; exit 1; }

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
    warn "Model not found at $MODEL"
    echo -n "Download it now? [Y/n]: "; read -r ans
    case "$ans" in
        n|N|no|No) exit 1 ;;
        *) download_model ;;
    esac
fi

# --- Validate VAD model ---
if [ "$VAD" = true ] && [ -z "$VAD_MODEL" ]; then
    err "--vad requires --vad-model PATH"
    exit 1
fi

# --- Find files ---
build_find_expr() {
    local expr=()
    for ext in "${EXTENSIONS[@]}"; do
        expr+=(-o -iname "*.$ext")
    done
    echo "${expr[@]:1}"
}

find_files() {
    local path="$1" resolved
    resolved="$(realpath -q "$path" 2>/dev/null || readlink -f "$path" 2>/dev/null || echo "$path")"
    if [ -f "$resolved" ]; then
        echo "$resolved"
    elif [ -d "$resolved" ]; then
        local depth_flag=""
        [ "$RECURSIVE" = false ] && depth_flag="-maxdepth 1"
        set -f
        # shellcheck disable=SC2086
        find "$resolved" $depth_flag -type f \( $(build_find_expr) \)
        set +f
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

# --- Check if all requested output formats already exist ---
output_exists() {
    local f="$1" out_base ext
    out_base=$(get_output_base "$f")
    ext="${f##*.}"
    IFS=',' read -ra fmts <<< "$FORMATS"
    for fmt in "${fmts[@]}"; do
        case "$fmt" in
            srt)  [ -f "$out_base.srt" ] || [ -f "$out_base.$ext.srt" ] || return 1 ;;
            txt)  [ -f "$out_base.txt" ] || [ -f "$out_base.$ext.txt" ] || return 1 ;;
            vtt)  [ -f "$out_base.vtt" ] || [ -f "$out_base.$ext.vtt" ] || return 1 ;;
            json) [ -f "$out_base.json" ] || [ -f "$out_base.$ext.json" ] || return 1 ;;
            tsv)  [ -f "$out_base.tsv" ] || [ -f "$out_base.$ext.tsv" ] || return 1 ;;
            lrc)  [ -f "$out_base.lrc" ] || [ -f "$out_base.$ext.lrc" ] || return 1 ;;
            csv)  [ -f "$out_base.csv" ] || [ -f "$out_base.$ext.csv" ] || return 1 ;;
        esac
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
        f="$tmp_wav"
    fi
    local -a cmd=("$WHISPER_BIN" -t "$THREADS" -f "$f" -m "$MODEL")

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

    set +e
    "${cmd[@]}"
    local rc=$?
    set -e
    [ -n "$tmp_wav" ] && rm -f "$tmp_wav"
    if [ "$rc" -eq 0 ] && [ -z "$OUTDIR" ]; then
        local old_suffix
        for old_suffix in srt txt vtt json tsv lrc csv; do
            [ -f "$out_base.$ext.$old_suffix" ] && rm -f "$out_base.$ext.$old_suffix"
        done
    fi
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
        local exit_code=$?
        echo -e "${RED}FAIL${NC}  $f (exit $exit_code)"
        log "FAIL  $f (exit $exit_code)"
        return "$exit_code"
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
[ "$DRY_RUN" = true ] && info "=== DRY RUN - no files will be transcribed ==="
echo ""

# --- Process ---
count_passed=0
count_failed=0
count_skipped=0
failed_files=()

if [ "$PARALLEL" -gt 1 ]; then
    info "Processing $total file(s) with $PARALLEL parallel job(s)..."

    temp_dir=$(mktemp -d)
    trap 'cleanup; rm -rf "$temp_dir"' EXIT
    export TEMP_DIR="$temp_dir"
    export WHISPER_BIN FFMPEG_CONVERT OUTDIR MODEL LANGUAGE THREADS FORMATS MIN_SIZE MAX_SIZE
    export SKIP_EXISTING QUIET LOG_FILE VAD VAD_MODEL TRANSLATE NO_GPU NO_FLASH_ATTN PRINT_PROGRESS
    export -f transcribe_one run_whisper output_exists get_output_base conv_to_wav NEEDS_CONVERT log info ok warn err
    export RED GREEN YELLOW CYAN NC

    printf '%s\0' "${all_files[@]}" | xargs -0 -P "$PARALLEL" -I{} bash -c '
        transcribe_one "$1"
        rc=$?
        h=$(printf "%s" "$1" | md5sum | cut -c1-12)
        echo "$rc" > "$TEMP_DIR/result_$h"
        [ "$rc" -gt 0 ] && [ "$rc" -ne 2 ] && echo "$1" > "$TEMP_DIR/fname_$h"
    ' _ {} 2>&1

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
    for f in "${all_files[@]}"; do
        code=0
        set +e
        transcribe_one "$f"
        code=$?
        set -e
        case "$code" in
            0) count_passed=$((count_passed + 1)) ;;
            2) count_skipped=$((count_skipped + 1)) ;;
            *) count_failed=$((count_failed + 1)); failed_files+=("$f") ;;
        esac
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
