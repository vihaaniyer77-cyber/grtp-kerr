#!/bin/bash
# Exit on any error
set -e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$HOME/Desktop/grtp_results_final"

# ---------------------------------------------------------------------------
# Progress bar utility
# ---------------------------------------------------------------------------
TOTAL_STEPS=4
PIPELINE_START=$(date +%s)

# Function to get cumulative progress fraction safely, agnostic of bash/zsh array indexing
get_cum_before() {
    local step=$1
    case "$step" in
        0) echo "0.00" ;;
        1) echo "0.45" ;;
        2) echo "0.85" ;;
        3) echo "0.97" ;;
        4) echo "1.00" ;;
        *) echo "0.00" ;;
    esac
}

_fmt_time() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $(( (s%3600)/60 )) $((s%60))
}

print_progress() {
    local step=$1        # 0 = starting, 1-4 = after completing that exp
    local label="$2"     # description of what is about to run
    local now
    now=$(date +%s)
    local elapsed=$(( now - PIPELINE_START ))
    local elapsed_fmt
    elapsed_fmt=$(_fmt_time "$elapsed")

    # Fraction of work DONE so far
    local frac_done
    frac_done=$(get_cum_before "$step")
    local frac_done_pct
    frac_done_pct=$(awk -v fd="$frac_done" 'BEGIN{printf "%.0f", fd * 100}')

    # ETA: if some work done, extrapolate total time safely
    local eta_str="calculating..."
    if [ "$elapsed" -gt 5 ] && [ "$step" -gt 0 ]; then
        local total_est
        total_est=$(awk -v fd="$frac_done" -v el="$elapsed" 'BEGIN { fd = fd + 0; if (fd <= 0) exit 1; printf "%.0f", el / fd }' 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$total_est" ]; then
            local remaining=$(( total_est - elapsed ))
            if [ "$remaining" -gt 0 ]; then
                eta_str=$(_fmt_time "$remaining")
            else
                eta_str="almost done"
            fi
        fi
    fi

    # Build the 30-char block bar
    local filled
    filled=$(awk -v fd="$frac_done" 'BEGIN{printf "%d", fd * 30}')
    local bar=""
    for ((i=0; i<30; i++)); do
        if   [ "$i" -lt "$filled" ]; then
            bar+="█"
        elif [ "$i" -eq "$filled" ] && [ "$step" -lt "$TOTAL_STEPS" ]; then
            bar+="▶"
        else
            bar+="░"
        fi
    done

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf  "║  GRTP-Kerr  [%s]  %3s%%                    ║\n" "$bar" "$frac_done_pct"
    printf  "║  Step %d / %d  ·  %s\n" "$step" "$TOTAL_STEPS" "$label"
    printf  "║  Elapsed: %s   ETA remaining: %s\n" "$elapsed_fmt" "$eta_str"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          GRTP-Kerr Simulation Pipeline — Full Run            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Project : %s\n" "$PROJECT_DIR"
printf "║  Results : %s\n" "$OUTPUT_DIR"
printf "║  Started : %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "╚══════════════════════════════════════════════════════════════╝"

mkdir -p "$OUTPUT_DIR/data"
mkdir -p "$OUTPUT_DIR/figures"

# Save Experiment 1 output files if they exist locally
echo "Checking and saving Experiment 1 results..."
if [ -f "data/exp1_efficiency.h5" ]; then
    cp data/exp1_efficiency.h5 "$OUTPUT_DIR/data/"
    echo "Saved data/exp1_efficiency.h5 to $OUTPUT_DIR/data/"
fi
if [ -f "figures/exp1_efficiency.pdf" ]; then
    cp figures/exp1_efficiency.pdf "$OUTPUT_DIR/figures/"
    echo "Saved figures/exp1_efficiency.pdf to $OUTPUT_DIR/figures/"
fi
if [ -f "figures/exp1_efficiency.png" ]; then
    cp figures/exp1_efficiency.png "$OUTPUT_DIR/figures/"
    echo "Saved figures/exp1_efficiency.png to $OUTPUT_DIR/figures/"
fi

cd "$PROJECT_DIR"

export USE_NUMBA=1

# Detect Python
if [ -f "$PROJECT_DIR/.venv/bin/python" ]; then
    VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
elif command -v uv &> /dev/null; then
    VENV_PYTHON="uv run python"
else
    VENV_PYTHON="python3"
fi

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

# Experiment 1 has already completed successfully (Total compute time: 52.3 min)
# print_progress 0 "Starting — Exp 1: Efficiency Sweep (a/M vs r_X heatmap)"
# $VENV_PYTHON experiments/exp1_efficiency_sweep.py "$@"

print_progress 1 "Exp 1 done — Exp 2: Basin-of-Fate Maps (fractal basins)"
$VENV_PYTHON experiments/exp2_basin_map.py "$@"

print_progress 2 "Exp 2 done — Exp 3: Lyapunov Exponents & Poincaré Sections"
$VENV_PYTHON experiments/exp3_lyapunov.py "$@"

print_progress 3 "Exp 3 done — Exp 4: Representative Trajectories & Diagnostics"
$VENV_PYTHON experiments/exp4_trajectories.py "$@"

# ---------------------------------------------------------------------------
# Copy results
# ---------------------------------------------------------------------------
print_progress 4 "All experiments complete — copying results..."
cp -r data/    "$OUTPUT_DIR/data/"
cp -r figures/ "$OUTPUT_DIR/figures/"

TOTAL_ELAPSED=$(( $(date +%s) - PIPELINE_START ))
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅  All 4 experiments completed successfully!               ║"
printf "║  Total wall time : %s                              ║\n" "$(_fmt_time $TOTAL_ELAPSED)"
printf "║  Results saved to: %s\n" "$OUTPUT_DIR"
echo "╚══════════════════════════════════════════════════════════════╝"
