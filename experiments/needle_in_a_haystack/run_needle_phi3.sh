# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

export TOKENIZERS_PARALLELISM=false

# Load Haystack
mkdir -p data
wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl

# Run the Needle in A Haystack Test
python experiments/needle_in_a_haystack/needle_test.py \
    --model_name microsoft/Phi-3-mini-128k-instruct \
    --max_length 131072 \
    --min_length 1024 \
    --rounds 5 \
    --attn_type hf \
    --output_path ./needle \
    --run_name hf_phi3 \
    --jobs 0-4

python experiments/needle_in_a_haystack/needle_test.py \
    --model_name microsoft/Phi-3-mini-128k-instruct \
    --max_length 131072 \
    --min_length 1024 \
    --rounds 5 \
    --attn_type hf \
    --kv_cache_cpu \
    --output_path ./needle \
    --run_name hf_phi3 \
    --jobs 4-15

# Data Summary
python experiments/needle_in_a_haystack/needle_summary.py --output_path ./needle --run_name hf_phi3

# Visualization
mkdir -p figures
python experiments/needle_in_a_haystack/needle_viz.py --res_file ./needle/hf_phi3.json --model_name phi3 --mode hf
