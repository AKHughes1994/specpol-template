#!/bin/bash
# Re-run only the preferred model(s) for each epoch in ../files/$SOURCE_DIR/QU_text
# (e.g. the models chosen by compare_models.py)

cd "$(dirname "$0")" || exit 1

# SOURCE_DIR (subdirectory of ../files containing QU_text/) is set in source_config.py
SOURCE_DIR=$(python3 -c "from source_config import SOURCE_DIR; print(SOURCE_DIR)") || exit 1

# =============================================================================
# SOURCE-SPECIFIC SETTINGS — review/change these when adapting to a new source
# =============================================================================
DEFAULT_MODELS="S P"             # used for any epoch not listed in BEST_MODELS

# Preferred model(s) per epoch. Key = data file stem, value = space-separated model types.
declare -A BEST_MODELS=(
#   ["<SOURCE_NAME>_QU_20240122"]="S"
#   ["<SOURCE_NAME>_QU_20240204_032245"]="SS ST"
)

for DATA in ../files/"$SOURCE_DIR"/QU_text/*.txt; do
    STEM=$(basename "$DATA" .txt)
    for MODEL in ${BEST_MODELS[$STEM]:-$DEFAULT_MODELS}; do
        echo "Running model $MODEL for $STEM..."
        python3 faraday_obs_fit.py "$MODEL" "$DATA"
    done
done
