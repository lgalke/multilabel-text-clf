#!/usr/bin/env bash
# Sweep encoder models (BERT-base, BERT-large, DistilBERT, RoBERTa) across datasets and seeds.
#
# Usage:
#   bash run_multilabel_encoders.bash
#
# Override defaults via environment variables:
#   DATA_ROOT=../multi_label_data  (root dir containing per-dataset folders)
#   OUTPUT_DIR=results             (where .json, .txt, _loss.png are written)
#   CUDA_VISIBLE_DEVICES=0         (GPU to use)
#   SEEDS="42 43 44"               (space-separated list of seeds)
#
# Example:
#   SEEDS="42 43 44" CUDA_VISIBLE_DEVICES=1 bash run_multilabel_encoders.bash

set -euo pipefail
SEEDS="100 101 102"
OUTPUT_DIR="results-encoders-v2"

#DATA_ROOT="${DATA_ROOT:-../multi_label_data}"
DATA_ROOT="/media/nvme4n1/project-textmlp/datasets"
OUTPUT_DIR="${OUTPUT_DIR:-results-encoders}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
read -ra SEEDS <<< "${SEEDS:-42}"
THRESHOLDS=(0.5 0.2)

# econbiz excluded for now
#DATASETS=(reuters amazon dbpedia goemotions)
DATASETS=(reuters rcv1-v2 amazon dbpedia nyt goemotions)

# model name (used for stem/json naming) -> script filename
declare -A SCRIPT_FOR
SCRIPT_FOR["bert-base"]="bert_model_multi_label.py"
SCRIPT_FOR["bert-large"]="bert_large_model_multi_label.py"
SCRIPT_FOR["distilbert"]="distilbert_model_multi_label.py"
SCRIPT_FOR["roberta"]="roberta_model_multi_label.py"

MODELS=(bert-base bert-large distilbert roberta)

mkdir -p "$OUTPUT_DIR" logs

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=== Sweep config ==="
echo "  DATA_ROOT  : $DATA_ROOT"
echo "  OUTPUT_DIR : $OUTPUT_DIR"
echo "  GPU        : $GPU"
echo "  SEEDS      : ${SEEDS[*]}"
echo "  DATASETS   : ${DATASETS[*]}"
echo "  MODELS     : ${MODELS[*]}"
echo "  THRESHOLDS : ${THRESHOLDS[*]}"
echo "===================="

for dataset in "${DATASETS[@]}"; do
    train_json="$DATA_ROOT/$dataset/train_data.json"
    test_json="$DATA_ROOT/$dataset/test_data.json"
    if [[ ! -f "$train_json" || ! -f "$test_json" ]]; then
        echo "[skip] $dataset — data not found ($train_json)"
        continue
    fi
    for model in "${MODELS[@]}"; do
        script="${SCRIPT_FOR[$model]}"
        for seed in "${SEEDS[@]}"; do
            stem="${model}_${dataset}_seed${seed}"
            done_all=true
            for thr in "${THRESHOLDS[@]}"; do
                [[ -f "$OUTPUT_DIR/${stem}_thr${thr}.json" ]] || { done_all=false; break; }
            done
            if $done_all; then
                echo "[skip] $stem — already done"
                continue
            fi
            echo "[run]  $stem"
            python "$script" \
                --dataset    "$dataset"    \
                --seed       "$seed"       \
                --data-root  "$DATA_ROOT"  \
                --output-dir "$OUTPUT_DIR" \
                --thresholds "${THRESHOLDS[@]}" \
                2>&1 | tee "logs/${stem}.log"
            echo "[done] $stem"
        done
    done
done

echo "=== Sweep complete. Results in $OUTPUT_DIR/ ==="
