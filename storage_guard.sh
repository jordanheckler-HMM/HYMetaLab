#!/usr/bin/env bash
# storage_guard.sh
# One-command storage control for your sim outputs.
# Usage examples:
#   chmod +x storage_guard.sh
#   ./storage_guard.sh --cap-gb 2 --keep-runs 5 bash scripts/phase2.sh
#   ./storage_guard.sh python -m analysis_fast.main --steps 30000 --epochs 5 --outputs outputs_p2_balanced

set -euo pipefail

############# CONFIG (override via flags) #############
CAP_GB="${CAP_GB:-3}"            # hard cap for total project outputs (in GB)
KEEP_RUNS="${KEEP_RUNS:-7}"      # keep most recent N run folders
OUT_ROOTS_DEFAULT="outputs outputs_fast outputs_ultrafast outputs_p2_fast outputs_p2_balanced outputs_p2_thorough outputs_phase2 discovery_results"
#######################################################

# ---- parse minimal flags ----
if [[ "${1:-}" == "--cap-gb" ]]; then CAP_GB="$2"; shift 2; fi
if [[ "${1:-}" == "--keep-runs" ]]; then KEEP_RUNS="$2"; shift 2; fi

CMD=("$@")
[[ ${#CMD[@]} -eq 0 ]] && {
  echo "storage_guard: No command provided. Example:"
  echo "  ./storage_guard.sh --cap-gb 2 --keep-runs 5 bash scripts/phase2.sh"
  exit 1
}

# ---- detect output roots present in repo ----
OUT_ROOTS=()
while IFS= read -r d; do
  [[ -d "$d" ]] && OUT_ROOTS+=("$d")
done < <(printf "%s\n" $OUT_ROOTS_DEFAULT | tr ' ' '\n')

# ---- helpers ----
bytes_to_gb () { awk -v b="$1" 'BEGIN{printf "%.2f", b/1024/1024/1024}'; }
du_total_bytes () { du -sb "${OUT_ROOTS[@]}" 2>/dev/null | awk '{s+=$1} END{print (s==""?0:s)}'; }

compress_images () {
  local dir="$1"
  # Prefer ImageMagick if available
  if command -v mogrify >/dev/null 2>&1; then
    # Strip metadata, cap size, and recompress
    find "$dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -size +200k -print0 \
      | xargs -0 -I{} mogrify -strip -resize '2000x2000>' -quality 70 "{}" || true
  elif command -v pngquant >/dev/null 2>&1; then
    find "$dir" -type f -iname "*.png" -size +200k -print0 \
      | xargs -0 -I{} pngquant --force --ext .png "{}" || true
  else
    # Fallback: leave images as-is
    :
  fi
}

zip_and_trim_runs () {
  local root="$1"
  [[ -d "$root" ]] || return 0

  # 1) Keep only key figures; nuke bulky non-essentials
  find "$root" -type f -path "*/figures/*" \
    ! -name "recurrence_plot_GLOBAL.png" \
    ! -name "dtw_heatmap.png" \
    ! -name "granger_heatmap.png" \
    ! -name "example_timeseries.png" \
    -delete 2>/dev/null || true

  # 2) Compress images that remain
  compress_images "$root"

  # 3) Remove huge CSVs (>50MB) but keep summaries
  find "$root" -type f -iname "*.csv" -size +50M \
    ! -name "*summary*.csv" \
    ! -name "combined_*.csv" \
    -delete 2>/dev/null || true

  # 4) Gzip bulky Markdown logs/reports > 2MB
  find "$root" -type f \( -iname "*.md" -o -iname "*.log" \) -size +2M -print0 \
    | xargs -0 -I{} bash -c 'gzip -f "{}" || true'

  # 5) Zip entire run subfolders and remove originals, keeping only latest N
  #    A "run subfolder" = direct child dir with timestamps/outputs.
  RUN_DIRS=()
  while IFS= read -r line; do
    RUN_DIRS+=("$(echo "$line" | awk '{print $2}')")
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -exec stat -f "%m %N" {} \; | sort -nr)
  local idx=0
  for rd in "${RUN_DIRS[@]}"; do
    ((idx++))
    # Keep latest KEEP_RUNS as folder; older → zip (if not already)
    if (( idx > KEEP_RUNS )); then
      if [[ ! -f "${rd}.zip" ]]; then
        zip -rq "${rd}.zip" "$rd" || true
        rm -rf "$rd"
      fi
    else
      # For the newest ones, ensure any nested heavy dirs are zipped
      if [[ -d "$rd/metrics_raw" ]]; then
        (cd "$rd" && zip -rq metrics_raw.zip metrics_raw && rm -rf metrics_raw) || true
      fi
    fi
  done
}

enforce_cap () {
  local cap_bytes
  cap_bytes=$(awk -v g="$CAP_GB" 'BEGIN{printf "%.0f", g*1024*1024*1024}')

  # First pass pruning across all roots
  for root in "${OUT_ROOTS[@]}"; do zip_and_trim_runs "$root"; done

  local used
  used=$(du_total_bytes)
  if (( used <= cap_bytes )); then
    echo "storage_guard: OK — using $(bytes_to_gb "$used") GB (cap ${CAP_GB} GB)."
    return 0
  fi

  # Aggressive prune: delete oldest zipped archives until under cap
  for root in "${OUT_ROOTS[@]}"; do
    OLD_ZIPS=()
    while IFS= read -r line; do
      OLD_ZIPS+=("$(echo "$line" | awk '{print $2}')")
    done < <(find "$root" -type f -name "*.zip" -exec stat -f "%m %N" {} \; | sort -n)
    for z in "${OLD_ZIPS[@]}"; do
      used=$(du_total_bytes)
      (( used <= cap_bytes )) && break
      rm -f "$z" || true
    done
  done

  used=$(du_total_bytes)
  echo "storage_guard: Post-prune usage: $(bytes_to_gb "$used") GB (cap ${CAP_GB} GB)."
  if (( used > cap_bytes )); then
    echo "storage_guard: WARNING — still above cap. Consider lowering KEEP_RUNS or CAP_GB, or moving archives off-disk."
  fi
}

# ---- pre-run prune (keeps workspace light) ----
[[ ${#OUT_ROOTS[@]} -gt 0 ]] && enforce_cap

# ---- run target command ----
echo "storage_guard: running → ${CMD[*]}"
"${CMD[@]}" || { echo "storage_guard: command failed"; exit 1; }

# ---- post-run prune/enforce cap ----
[[ ${#OUT_ROOTS[@]} -gt 0 ]] && enforce_cap

# ---- friendly summary ----
used_now=$(du_total_bytes)
echo "storage_guard: done. Current usage: $(bytes_to_gb "$used_now") GB (cap ${CAP_GB} GB), keeping ${KEEP_RUNS} newest runs per output root."