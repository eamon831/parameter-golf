# Experiment Log

All experiments tracked here. Never re-run a failed experiment without a new hypothesis.

---

## Experiment 0: Reproduce SOTA baseline
- Date: pending (waiting for RunPod credits, applied 2026-03-28)
- Hypothesis: SOTA code (PR #549) runs as documented and produces ~1.1194 BPB
- Change: none — run SOTA train_gpt.py unmodified
- Hardware: 8xH100 SXM (required for Parallel Muon)
- Command: `SEED=1337 RUN_ID=exp0_reproduce torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- Expected: val_bpb ~1.1192 (seed 1337 from their logs), ms/step ~83ms, artifact ~15.97MB
- Result: pending
- Verdict: pending
- Cost: ~$3.30 (10 min training + 10 min eval on 8xH100 @ ~$20/hr)

## Experiment 0.5: Verify our code matches SOTA (JEPA off)
- Date: pending
- Hypothesis: our_train_gpt.py with JEPA_ENABLED=0 produces identical results to original SOTA
- Change: use our code instead of original — should be identical since JEPA defaults to off
- Hardware: 8xH100 SXM
- Command: `SEED=1337 JEPA_ENABLED=0 RUN_ID=exp05_ours_baseline torchrun --standalone --nproc_per_node=8 our_train_gpt.py`
- Expected: val_bpb ~1.1192 (same as Experiment 0)
- Result: pending
- Verdict: pending

## Experiment 1: JEPA auxiliary loss
- Date: pending
- Hypothesis: JEPA auxiliary loss improves BPB by acting as regularizer (PR #1006 claims it contributes to 1.1085). Extra ~130K params + forward compute for JEPA loss should add <5ms/step.
- Change: JEPA_ENABLED=1 (latent_dim=256, loss_weight=0.12, future_spans=1,2,4,8, ema_decay=0.996)
- Hardware: 8xH100 SXM
- Command: `SEED=1337 JEPA_ENABLED=1 RUN_ID=exp1_jepa torchrun --standalone --nproc_per_node=8 our_train_gpt.py`
- Success criteria: BPB improves by >0.001 AND ms/step increases by <5ms
- Fail criteria: BPB worsens OR ms/step increases by >5ms (net negative per Rule 1)
- Result: pending
- Verdict: pending

## Experiment 2: Full Hessian GPTQ (planned, not yet coded)
- Date: pending
- Hypothesis: Full GPTQ (Frantar et al.) gives better int6 quantization than GPTQ-lite by compensating per-column rounding error using inverse Hessian. PR #1006 reports 13s runtime.
- Change: Replace GPTQ-lite quantization with full Hessian GPTQ + 128-batch calibration
- Hardware: 8xH100 SXM
- Success criteria: Post-quantization BPB improves (lower quantization penalty)
- Code status: NOT YET IMPLEMENTED. Reference implementation in PR #1006.
- Result: pending

## Experiment 3: AdamW TTT pre-quantization (planned, not yet coded)
- Date: pending
- Hypothesis: AdamW TTT on full-precision EMA weights before quantization gives better adaptation than SGD TTT on dequantized weights. PR #1006 found SGD fails on CastedLinear.
- Change: Replace SGD TTT with AdamW + cosine decay, run before GPTQ instead of after
- Hardware: 8xH100 SXM
- Success criteria: TTT BPB gain > current 0.0025 from SOTA's SGD TTT
- Code status: NOT YET IMPLEMENTED. Reference implementation in PR #1006.
- Result: pending

---

## Research Notes (no experiment needed)

### 8192 Vocab Analysis (2026-03-28)
- INVESTIGATED: Can we fit 8192 vocab in the SOTA's 16MB budget?
- ANSWER: No. SOTA artifact has only 9,994 bytes headroom. Even factored 8192×64 + 64×512 needs +25KB.
- Only viable with ternary quantization (4x more compact) which is a completely different codebase.
- DECISION: Stay on int6/GPTQ path. Don't pursue 8192 vocab.

### Depth Recurrence (2026-03-28)
- CONFIRMED DEAD END by 3 independent teams (PR #363, PR #499, ternary author)
- Two structural taxes: quantization compounding + step time overhead = 0.025 BPB worse
- Do not attempt.

### Novel Architectures (2026-03-28, PR #831)
- 6 architectures tested, all failed at 16MB/600s constraint
- SOTA stack is co-optimized: Parallel Muon + torch.compile + int6 + H100 tensor cores
- Throughput tax: must improve BPB by 0.007 per ms/step overhead
- SSM hybrid (Mamba) is 3.4x slower — completely unviable
