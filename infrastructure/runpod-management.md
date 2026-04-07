# RunPod Management for Parameter Golf

**Date:** 2026-04-07
**Context:** Managing H100 pods for training experiments. Budget is tight ($25 initial grant), so start/stop discipline is critical.

## Auth & Base URL

```
Base: https://rest.runpod.io/v1/
Header: Authorization: Bearer $RUNPOD_API_KEY
```

API key is in `~/.zshrc`. Same key handles pods + billing.

## Pod Lifecycle — The Pattern

```
Create pod (once) → Start → SSH in → Run experiment → Stop (billing stops)
                     ↑                                      |
                     └──────── next experiment ─────────────┘
```

`/workspace` persists across stop/start. Dataset, repo, and pip packages survive. Only pay while RUNNING.

## Key API Commands

### List Pods
```bash
curl -s https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -m json.tool
```

### Create 8xH100 Pod (Parameter Golf Template)
```bash
curl -X POST https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "imageName": "runpod/pytorch:2.9.1-py3.12-cuda12.8.1-cudnn9.8.0-devel-ubuntu22.04",
    "name": "param-golf-8xh100",
    "cloudType": "SECURE",
    "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
    "gpuCount": 8,
    "containerDiskInGb": 100,
    "volumeInGb": 50,
    "ports": ["22/tcp"],
    "env": {}
  }'
```

Or use the official template link:
```
https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th
```

### Start Pod
```bash
curl -X POST https://rest.runpod.io/v1/pods/{podId}/start \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Stop Pod (CRITICAL — do this immediately after experiment)
```bash
curl -X POST https://rest.runpod.io/v1/pods/{podId}/stop \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Get Pod Status
```bash
curl -s https://rest.runpod.io/v1/pods/{podId} \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -m json.tool
```

### Delete Pod (permanent — only when done with all experiments)
```bash
curl -X DELETE https://rest.runpod.io/v1/pods/{podId} \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

### Check Billing
```bash
curl -s https://rest.runpod.io/v1/billing/pods \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | python3 -m json.tool
```

## Cost Awareness

| Config | Cost/hr | Cost per 10-min experiment |
|--------|---------|---------------------------|
| 1xH100 | ~$2.50 | ~$0.42 |
| 8xH100 | ~$20.00 | ~$3.33 |

**$25 budget:**
- ~60 runs on 1xH100 (directional testing)
- ~7 runs on 8xH100 (final validation)

**Rules:**
1. Always STOP pod after experiment finishes
2. Use 1xH100 for directional testing first
3. Only use 8xH100 for final 3-seed validation
4. Never leave a pod running overnight

## First-Time Pod Setup (run once after create)

```bash
cd /workspace
git clone https://github.com/eamon831/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install --break-system-packages sentencepiece zstandard
```

This data persists in `/workspace` across stop/start.

## Running Experiments

### Experiment 0: Reproduce SOTA baseline
```bash
cd /workspace/parameter-golf
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 RUN_ID=exp0_baseline \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py
```

### Experiment 1: JEPA enabled
```bash
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 RUN_ID=exp1_jepa JEPA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py
```

### After each experiment: STOP THE POD
```bash
# From local Mac:
curl -X POST https://rest.runpod.io/v1/pods/{podId}/stop \
  -H "Authorization: Bearer $RUNPOD_API_KEY"
```

## Quick Reference Script

Save as `runpod.sh` in the repo root:

```bash
#!/bin/bash
# Usage: ./runpod.sh [list|start|stop|status|billing]
API_KEY="${RUNPOD_API_KEY}"
BASE="https://rest.runpod.io/v1"
POD_ID="${RUNPOD_POD_ID}"  # Set after first create

case "$1" in
  list)    curl -s "$BASE/pods" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool ;;
  start)   curl -X POST "$BASE/pods/$POD_ID/start" -H "Authorization: Bearer $API_KEY" ;;
  stop)    curl -X POST "$BASE/pods/$POD_ID/stop" -H "Authorization: Bearer $API_KEY" ;;
  status)  curl -s "$BASE/pods/$POD_ID" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool ;;
  billing) curl -s "$BASE/billing/pods" -H "Authorization: Bearer $API_KEY" | python3 -m json.tool ;;
  *)       echo "Usage: $0 [list|start|stop|status|billing]" ;;
esac
```

## Lesson from studio-ops

> **Rule: Stop pods immediately after use.** A pod left running for 1 hour at 8xH100 costs $20 — that's 6 experiments wasted. Set a phone timer if you need to.
