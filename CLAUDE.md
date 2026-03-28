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

## Current Leaderboard Context (2026-03-28)

- SOTA: 1.1194 BPB (LeakyReLU² + TTT + Parallel Muon)
- Baseline: 1.2244 BPB
- Gap: 0.105 BPB across ~20 submissions in 10 days

## Strategy

Based on research (rules above), our highest-ROI path:

1. **Start from SOTA code** (rule 4) — don't rebuild
2. **Explore larger vocab** (rule 3) — 8192 BPE is the biggest untapped lever in the SOTA path
3. **Factored embeddings** to fit larger vocab in 16MB budget
4. **Keep everything that works** — LeakyReLU², TTT, Parallel Muon, EMA+SWA, XSA, BigramHash, GPTQ-lite
5. **Ablate each change** (rule 6) — measure before and after
6. **Never add ms/step without measuring** (rule 1)

## Workflow

- Develop locally, don't push until ready
- Test on Mac (MLX) for quick iteration on architecture changes
- Run real benchmarks on RunPod (H100) for timing + BPB
- Submit PR to openai/parameter-golf when competitive

## Remotes

- `origin` → eamon831/parameter-golf (fork)
- `upstream` → openai/parameter-golf (submit PRs here)
