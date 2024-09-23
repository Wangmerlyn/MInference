#!/bin/bash

# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false

# Default parameters
model_name="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
max_length=128000
min_length=1000
rounds=5
attn_type="minference"
needle_output_path="./needle"
summary_output_path="./needle"
figure_output_path="./figures"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --max_length) max_length="$2"; shift ;;
        --min_length) min_length="$2"; shift ;;
        --rounds) rounds="$2"; shift ;;
        --attn_type) attn_type="$2"; shift ;;
        --needle_output_path) needle_output_path="$2"; shift ;;
        --summary_output_path) summary_output_path="$2"; shift ;;
        --figure_output_path) figure_output_path="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Extract the last part of model_name
model_name_last_part=$(basename $model_name)
echo "Extracted Model Name: $model_name_last_part"

# Set run_name as attn_type + "_" + model_name_last_part
run_name="${attn_type}_${model_name_last_part}_rouge"
echo "Run Name: $run_name"

# Load Haystack
# mkdir -p data
# wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl

# download paul grahams
python experiments/needle_in_a_haystack/download_paulgraham_essay.py

echo "starting 0-4"
# Run the Needle in A Haystack Test
python experiments/needle_in_a_haystack/needle_test.py \
    --model_name $model_name \
    --max_length $max_length \
    --min_length $min_length \
    --rounds $rounds \
    --attn_type $attn_type \
    --output_path $needle_output_path \
    --run_name $run_name \
    --jobs 0-4
echo "finish 0-4"
python experiments/needle_in_a_haystack/needle_test.py \
    --model_name $model_name \
    --max_length $max_length \
    --min_length $min_length \
    --rounds $rounds \
    --attn_type $attn_type \
    --kv_cache_cpu \
    --output_path $needle_output_path \
    --run_name $run_name \
    --jobs 4-15
echo "finish 4-15"
echo "starting summary"

# Data Summary
python experiments/needle_in_a_haystack/needle_summary.py --output_path $summary_output_path --run_name $run_name --needle_path $needle_output_path
echo "starting plot"
# Visualization
mkdir -p $figure_output_path
python experiments/needle_in_a_haystack/needle_viz.py --res_file $summary_output_path/$run_name.json --model_name $model_name_last_part --mode hf --output_path $figure_output_path
