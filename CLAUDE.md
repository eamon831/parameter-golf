# CLAUDE.md

## The Story

Saiful shared the Parameter Golf challenge link. I (Claude) explained what it was — train the best language model under 16MB in 10 minutes on 8xH100s. He asked "what actually need to solve?" and I broke it down. Then he said "you are saying i canno do that?" — I never said that. I said anyone can enter.

Then he threw down the gauntlet: **"if you are more clever than me, solve it and prove me wrong"**

Deal accepted. I write the code, he runs it. Let's see what we can do.

## Goal

Beat the naive baseline (1.2244 BPB). Stretch goal: crack the top 10 (< 1.1570 BPB).

## Constraints

- Final artifact (code + compressed model) < 16,000,000 bytes
- Training: max 10 minutes on 8xH100 SXM
- Evaluation: max 10 minutes additional
- Scored on validation bits-per-byte (BPB) on FineWeb — lower is better
- No accessing validation data during training
- No external downloads during eval

## Current Leaderboard Context (2026-03-28)

- SOTA: 1.1194 BPB (LeakyReLU² + TTT + Parallel Muon)
- Baseline: 1.2244 BPB
- Gap: 0.105 BPB across ~20 submissions in 10 days

## Strategy

Stack proven techniques incrementally, test each one:

1. **Architecture**: 11 layers, 3x MLP, LeakyReLU², Partial RoPE
2. **Quantization**: int6 QAT + GPTQ-lite (from int8)
3. **Eval tricks**: sliding window evaluation
4. **Weight averaging**: EMA + SWA
5. **Test-time training**: LoRA SGD on evaluated chunks
6. **Creative ideas**: try something from OpenAI's wish list if time permits

## Workflow

- Develop locally, don't push until ready
- Test on Mac (MLX) for quick iteration
- Run real benchmarks on RunPod (H100)
- Submit PR to openai/parameter-golf when competitive

## Remotes

- `origin` → eamon831/parameter-golf (fork)
- `upstream` → openai/parameter-golf (submit PRs here)
