#!/bin/bash
cd /workspace/parameter-golf
mkdir -p logs
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export WARMDOWN_ITERS=4000
export TARGET_MB=15.9
export SEED=1337
export RUN_ID=exp1_jepa_8x3
export JEPA_ENABLED=1
nohup torchrun --standalone --nproc_per_node=8 our_train_gpt.py > logs/exp1_jepa_8x3.txt 2>&1 &
echo "Started! Monitor with: tail -f logs/exp1_jepa_8x3.txt"
