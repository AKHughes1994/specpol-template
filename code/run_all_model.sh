#!/bin/bash
# Fit a set of models to every epoch in ../files/$SOURCE_DIR/QU_text

cd "$(dirname "$0")" || exit 1

# SOURCE_DIR (subdirectory of ../files containing QU_text/) is set in source_config.py
SOURCE_DIR=$(python3 -c "from source_config import SOURCE_DIR; print(SOURCE_DIR)") || exit 1

# =============================================================================
# SOURCE-SPECIFIC SETTINGS — review/change these when adapting to a new source
# =============================================================================
MODELS="S SS SSS T ST P PP"    # model types fitted to every epoch

# Optional: extra (more complex) models for specific epochs.
# Key = substring of the data file stem, value = space-separated model types.
declare -A EXTRA_MODELS=(
#   ["20240204_032245"]="SSSS SST STT"
)

for MODEL in $MODELS; do
    echo "Running model $MODEL..."
    for DATA in ../files/"$SOURCE_DIR"/QU_text/*.txt; do
        python3 faraday_obs_fit.py "$MODEL" "$DATA"
    done
done

for KEY in "${!EXTRA_MODELS[@]}"; do
    for DATA in ../files/"$SOURCE_DIR"/QU_text/*"$KEY"*.txt; do
        for MODEL in ${EXTRA_MODELS[$KEY]}; do
            echo "Running extra model $MODEL for $DATA..."
            python3 faraday_obs_fit.py "$MODEL" "$DATA"
        done
    done
done
