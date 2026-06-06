#!/usr/bin/env bash
# Run WideMLP on the Web of Science (WoS) dataset.
# WoS-yy/wos/ must contain train.src, train.tgt, val.src, val.tgt, test.src, test.tgt
# Adjust DATASET_FOLDER to the absolute or relative path of the WoS-yy directory.

MODEL_TYPE="mlp"
TOKENIZER_NAME="bert-base-uncased"
DATASET_FOLDER="../WoS-yy"
DATASET="wos"
BATCH_SIZE=128
EPOCHS=100
RESULTS_FILE="results_mlp_wos.csv"

set -e

for agg in mean tfidf; do
    for seed in 1 2 3 4 5; do
        python3 run_text_classification.py \
            --dataset_folder "$DATASET_FOLDER" \
            --model_type "$MODEL_TYPE" \
            --tokenizer_name "$TOKENIZER_NAME" \
            --bow_aggregation "$agg" \
            --threshold 0.2 0.5 \
            --batch_size "$BATCH_SIZE" \
            --epochs "$EPOCHS" \
            --num_workers 4 \
            --results_file "$RESULTS_FILE" \
            "$DATASET"
    done
done
