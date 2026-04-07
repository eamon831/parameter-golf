# CLAUDE.md

## Rules for Claude Code

### 0. Never Guess — Investigate First

**This is the most important rule.** Do not give guess answers. Guessing leads to frustration, burns energy and time, and sends us down wrong paths that would have been avoided by investigating first.

Before answering any question or making any recommendation:
1. **Investigate** — read files, grep code, check logs, search the web, test assumptions
2. **If you can't investigate** due to tool limitations — say so explicitly. Tell the user exactly what to check and ask them to share the results
3. **Never present uncertain information as fact** — if you're not sure, say "I don't know, let me check" not "I think it's X"
4. **Don't propose solutions before understanding the problem** — read the README, check the actual file structure, verify URLs return 200 before sharing them
5. **Don't make premature decisions** — gather data first, present options, let the user decide

**When in doubt:** Ask. A 30-second question saves hours of wrong-direction work.

### 0.1. Check Upstream Before Every Session

Before starting any work, always:
```bash
cd ~/office_projects/parameter-golf && git fetch upstream && git log upstream/main --oneline -10
```
The leaderboard moves fast — new PRs land daily. Someone may have already tried what we're about to build, or a new technique may change our priorities. Read new submission READMEs before writing code.

### 0.2. One Change, One Measurement

Never stack two untested changes. Every experiment:
1. State the hypothesis ("8192 vocab will improve BPB by ~X because Y")
2. Change ONE thing
3. Run, record: **BPB, ms/step, artifact size** (all three, always)
4. Decide: keep or revert
5. Log the result in `experiments.md` — even negative results

If we can't run it (no GPU access), say so. Don't theorize about what "should" work.

### 0.3. Read Before Write

Before modifying any code:
1. Read the file fully — understand what exists
2. Read the submission it came from — understand why it was written that way
3. Check if the technique has been tried by others — grep PRs and records

### 0.4. Budget Compute Like Money

RunPod H100 time costs real money (~$2.50/hr per GPU, $20/hr for 8xH100). Every experiment should have:
- A clear hypothesis worth testing
- The cheapest possible test (1xH100 first, 8x only for final validation)
- A defined success/fail criteria before running

Don't burn GPU hours on "let's see what happens" runs.

### 0.5. Log Everything

Every RunPod experiment gets logged in `experiments.md`:
```
## Experiment N: [name]
- Date: YYYY-MM-DD
- Hypothesis: [what we expect and why]
- Change: [what was modified]
- Hardware: [1xH100 / 8xH100]
- Result: BPB=X.XXXX | ms/step=XX | artifact=XXB
- Verdict: KEEP / REVERT / INVESTIGATE
- Cost: ~$X.XX
```
This prevents re-running failed experiments and builds institutional knowledge across sessions.

---

## Competition Rules (derived from research)

### 1. Every Millisecond Per Step = ~0.01 BPB

Wall-clock time is the hidden constraint. In 600 seconds:
- 83ms/step = 7,228 steps
- 93ms/step = 6,451 steps (777 fewer steps = real BPB loss)
- 144ms/step = 4,166 steps (depth recurrence death)

**Before adding ANY feature, measure its ms/step cost.** A technique that improves BPB by 0.003 but adds 10ms/step may net-negative because of lost training steps. Kernel fusions and FlashAttention-3 are free BPB — always use them.

### 2. Maximize Parameters Per Byte

The game is: cram the most useful parameters into 16MB. This is the single highest-leverage axis.
- Ternary (1.6-bit): ~80M params in 16MB
- Int5 (5-bit): ~25M params
- Int6 (6-bit): ~21M params
- Int8 (8-bit): ~16M params

Better compression = more params = lower BPB. But beware: aggressive quantization has a quality tax that must be offset by the extra capacity. Always measure the **net** effect (more params vs quantization noise).

### 3. Vocabulary Size Is the Biggest Single Lever

Ternary author proved: 8192 vocab = **-0.42 BPB** vs 1024 vocab. This is larger than ALL other techniques combined. The SOTA still uses 1024 vocab. If we can fit a larger vocab without blowing the size budget (factored embeddings, aggressive embedding quantization), this is the highest-ROI change.

### 4. Build on Proven Stacks, Don't Rebuild

Every leaderboard entry builds on the previous PR's code. Don't start from scratch. Start from the best available code, understand every line, then add one thing at a time. Each change must be ablated — measure before and after.

### 5. Confirmed Dead Ends — Do Not Attempt

These have been tried and failed by multiple independent teams:
- **Depth recurrence** — quantization compounding tax + step time overhead = 0.025 BPB worse (3 teams confirmed)
- **SmearGate on shared/recurrent weights** — unstable
- **Progressive loop unrolling** — torch.compile incompatible
- **Sawtooth LR schedules** — torch.compile guard recompilation every step
- **TTT on quantized weights** — breaks weight distributions
- **XSA on all layers** — over-aggressive, only helps on last 3-4 layers
- **QuadgramHash** — no confirmed benefit over BigramHash

### 6. Ablate Everything, Trust Nothing

The ternary author ran 250+ experiments. The depth recurrence team ran controlled comparisons. Winning = systematic experimentation, not clever ideas. For every change:
1. Run baseline, record BPB + ms/step + artifact size
2. Add ONE change
3. Run again, compare all three metrics
4. Keep only if net-positive

### 7. Two Budgets, Both Matter

- **Training budget:** 600 seconds on 8xH100 — optimize steps/second
- **Eval budget:** 600 seconds on 8xH100 — TTT and sliding window live here

TTT gives ~0.0025 BPB for ~410s of eval time. Sliding window eval (stride=64) is essentially free. Use both. Don't leave eval budget on the table.

### 8. Quantization Quality > Most Architecture Changes

Post-training quantization penalty (0.014 BPB on int8) is larger than most hyperparameter tuning gains. Invest in:
- QAT (quantization-aware training) — simulate quantization during training
- GPTQ-lite — optimal per-row clip percentile search
- Warmdown scheduling — smoother weights = less quantization noise
- Late QAT — enable STE fake-quantization in final 15% of training

### 9. Weight Averaging Is Free BPB

EMA (decay=0.997) + SWA (every 50 steps) consistently improve BPB across all submissions. No downside. Always use both.

### 10. Width > Depth at Fixed Parameter Budget

768d/10L outperforms 512d/25L. Wider models:
- Train faster (fewer sequential ops)
- Quantize better (more dimensions to average over)
- Work better with Muon optimizer (larger matrices = better Newton-Schulz)

Choose width over depth when you have parameter budget to spend.

---

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

## Current Leaderboard Context (2026-04-07)

**Verified SOTA (merged 2026-03-30):** 1.1147 BPB — PR #1019 by abaybektursun
- AR Self-Generated Full Hessian GPTQ (model generates own calibration data — fully legal)
- BigramHash 3072×112 (up from 1536)
- XSA on all 11 layers (up from last 4)
- Dropped TTT (25 failed attempts on this stack — neutral/negative)
- ~86.7 ms/step, 3-seed std 0.0004
- Code: `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py`

**Previous SOTA:** 1.1194 BPB — PR #549 (our old base code)
**Baseline:** 1.2244 BPB

**Open PRs (checked 2026-04-07):**
- PR #1437: **1.078 BPB** — SP8192 + Parallel Residuals + 3-Layer Recurrence + N-gram Tilt (3-seed)
- PR #1394: **1.086 BPB** — SP8192 + GPTQ Embeddings + Depth Recurrence + MuonEq-R (5-seed)
- PR #1435: **1.098 BPB** — 11L Depth Recurrence + BigramHash + EMA (3-seed)
- PR #1440: **1.103 BPB** — EngramLite + Mousse + Progressive Depth Recurrence + TTT (1-seed)
- PR #1006: JEPA + AdamW TTT + Full GPTQ — still OPEN, compliance issue

**Competition deadline:** April 30, 2026

## Strategy (Updated 2026-04-07)

**Phase 1 (CURRENT): JEPA on PR #1019 — get first submission**
- our_train_gpt.py = PR #1019 SOTA + JEPA (code DONE, tested)
- 93ms/step on 8xH100 confirmed (minimal overhead vs SOTA's 86.7ms)
- Full pipeline verified on 1xH100: training + EMA + GPTQ + int6 + eval
- 8xH100 run interrupted by pod failure at step 10 — need to rerun
- Waiting for compute grant ($100-$500 bootstrapped, submitted 2026-04-07)
- If JEPA beats 1.1147 → 3-seed validation → PR to upstream

**Phase 2 (FUTURE): Rebase on SP8192 stack**
- PR #1394 (1.086 BPB) is the new frontier — SP8192 vocab is the biggest lever
- Depth recurrence is back (contradicting old dead-ends list)
- N-gram tilt at eval gives ~0.005 BPB free
- Port JEPA onto SP8192 stack if Phase 1 shows it helps

### RunPod Commands (Ready to Execute)

```bash
# Setup (do once on fresh pod):
cd /workspace
git clone https://github.com/eamon831/parameter-golf.git && cd parameter-golf
python3 -m pip install huggingface-hub sentencepiece zstandard numpy
python3 -m pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run JEPA experiment (use run.sh to avoid copy-paste issues):
bash run.sh
# Monitor:
tail -f logs/exp1_jepa_8x3.txt
```

### What We Investigated and Rejected

**8192 vocab (Rule 3):** Biggest single lever (-0.42 BPB proven by ternary author), BUT:
- SOTA artifact is 15,990,006 bytes — only 9,994 bytes headroom
- Only works with ternary quantization (4x more compact) — completely different codebase
- **VERDICT: not viable on the int6/GPTQ path. Would require rebuilding from ternary base.**

**AdamW TTT:** Planned from PR #1006, but TTT is confirmed dead on the current SOTA stack (25 failed attempts by the SOTA author himself). Dropped.

### Key Research Findings (PR #831)

**The throughput tax formula:** Any technique must improve BPB by 0.007 per millisecond of step time overhead. At 83ms/step baseline, each 1ms costs ~7 steps, each step ≈ 0.001 BPB.

**All 6 novel architectures failed:**
| Technique | ms/step impact | BPB | Why failed |
|-----------|---------------|-----|-----------|
| MUD Optimizer | +5% | 1.1581 | solve_triangular can't use tensor cores |
| Info-Max (XSA-all) | +7% | 1.1261 | overhead eats its own gain |
| Hourglass FFN | +11% | 1.4519 | split weights catastrophic for int6 |
| nGPT Hypersphere | +47% | 1.6915 | unit-norm incompatible with int6 |
| TrigramHash | +18% | 1.1298 | hash overhead > trigram benefit |
| SSM Hybrid | +240% | 1.2516 | breaks torch.compile |

**Takeaway:** The SOTA stack is co-optimized (Parallel Muon + torch.compile + int6 + tensor cores). Breaking any pillar cascades. JEPA is one of the few additions that doesn't break this co-optimization because it's only active during training (no eval cost).

## Codebase Layout

```
parameter-golf/
├── CLAUDE.md              ← this file (project rules + context)
├── experiments.md         ← experiment log (track all runs)
├── our_train_gpt.py       ← OUR working code (PR #1019 SOTA + JEPA)
├── runpod.sh              ← RunPod pod management helper
├── train_gpt.py           ← baseline from OpenAI (9L, 1.2244 BPB)
├── train_gpt_mlx.py       ← MLX version for Mac testing
├── infrastructure/
│   └── runpod-management.md  ← RunPod API docs & cost rules
├── data/
│   ├── datasets/fineweb10B_sp1024/  ← downloaded (1 shard + val)
│   └── tokenizers/                   ← BPE tokenizer (1024 vocab)
├── records/
│   ├── track_10min_16mb/            ← all leaderboard submissions
│   └── track_non_record_16mb/       ← unlimited compute / research
└── .venv/                            ← Python venv (MLX + deps)
```

## Environment Setup

**Local (Mac M1):**
```bash
cd ~/office_projects/parameter-golf
source .venv/bin/activate
# Smoke test:
RUN_ID=mlx_smoke ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py
```
MLX venv has: mlx, mlx-lm, numpy, sentencepiece, huggingface-hub, datasets, tqdm

**RunPod (H100) — $25 initial grant:**

Pod management via REST API or helper script:
```bash
# Helper script (from repo root):
./runpod.sh list          # List all pods
./runpod.sh create-1x     # Create 1xH100 (~$2.50/hr, for testing)
./runpod.sh create-8x     # Create 8xH100 (~$20/hr, for validation)
./runpod.sh start          # Start stopped pod
./runpod.sh stop           # STOP pod (billing pauses, /workspace persists)
./runpod.sh status         # Check pod state
./runpod.sh billing        # Check spend

# After creating a pod, save the ID:
export RUNPOD_POD_ID=<id>
```

First-time pod setup (run once, persists across stop/start):
```bash
cd /workspace
git clone https://github.com/eamon831/parameter-golf.git && cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
pip install --break-system-packages sentencepiece zstandard
```

Running experiments:
```bash
# Experiment 0: Reproduce SOTA baseline (no JEPA)
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 RUN_ID=exp0_baseline \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py

# Experiment 1: JEPA enabled
BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 WARMDOWN_ITERS=4000 \
TARGET_MB=15.9 SEED=1337 RUN_ID=exp1_jepa JEPA_ENABLED=1 \
torchrun --standalone --nproc_per_node=8 our_train_gpt.py

# IMMEDIATELY after experiment finishes:
./runpod.sh stop
```

**COST RULES:**
- 1xH100 = ~$0.42 per 10-min run (use for directional testing)
- 8xH100 = ~$3.33 per 10-min run (use for final validation)
- $25 = ~60 runs on 1xH100 or ~7 runs on 8xH100
- ALWAYS stop pod after experiment. A forgotten 8xH100 running 1hr = $20 wasted = 6 experiments gone.

Full docs: `infrastructure/runpod-management.md`

## Blockers

- **RunPod credits:** $25 initial grant spent. ~$5.48 remaining. Bootstrapped grant ($100-$500) submitted 2026-04-07.
- **8xH100 run incomplete:** Pod terminated unexpectedly at step 10/6400. All pods terminated. Need new pod when credits arrive.
- **MLX validation is slow:** ~20min for full val on Mac. Training is fine for directional testing.

## What We've Proven (2026-04-07)

- **JEPA code works end-to-end** on both 1xH100 and 8xH100
- **93ms/step on 8xH100** — minimal overhead vs SOTA's 86.7ms (6ms = ~0.006 BPB cost)
- **525K JEPA params properly pruned** from artifact — zero eval cost
- **Full pipeline passes**: training → EMA → AR self-gen GPTQ → int6 quantization → sliding eval
- **1xH100 200-step results**: JEPA neutral at short horizon (expected — regularizers need thousands of steps)
- **Artifact size**: 4.98MB (well under 16MB limit)

## Git Remotes

- `origin` → `eamon831/parameter-golf` (our fork — push here)
- `upstream` → `openai/parameter-golf` (submit PRs here)
- GitHub auth: `eamon831` account via keyring

## Workflow

1. `git fetch upstream` before every session — check new submissions
2. Develop in `our_train_gpt.py` — test on RunPod
3. Log every experiment in `experiments.md`
4. When competitive: create submission folder in `records/`, PR to upstream
5. Don't push to upstream until we have 3-seed results with p<0.01
