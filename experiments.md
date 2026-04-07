# Experiment Log

All experiments tracked here. Never re-run a failed experiment without a new hypothesis.

---

## Experiment 0: Smoke test on 1xH100 (baseline, 50 steps)
- Date: 2026-04-07
- Hypothesis: Our code (PR #1019 base, JEPA off) runs correctly on H100
- Change: none — JEPA_ENABLED=0 (default)
- Hardware: 1xH100 SXM ($2.69/hr)
- Result: val_bpb=3.3678 (50 steps) | ms/step=648 | artifact=4,598,403 bytes
- Post-EMA val_bpb=4.0261 | final_int6 val_bpb=4.0347
- Verdict: PASS — code works, pipeline complete
- Cost: ~$0.15

## Experiment 1a: JEPA on 1xH100 (200 steps)
- Date: 2026-04-07
- Hypothesis: JEPA adds minimal overhead and produces valid results
- Change: JEPA_ENABLED=1 (latent_dim=256, loss_weight=0.12, spans=1,2,4,8, ema_decay=0.996)
- Hardware: 1xH100 SXM
- Result: val_bpb=2.7733 | ms/step=711 | artifact=4,982,147 bytes
- Post-EMA val_bpb=3.3727 | final_int6 val_bpb=3.4159
- JEPA params excluded from export: 525,312
- Verdict: PASS — full pipeline works with JEPA
- Cost: ~$0.50

## Experiment 1b: Baseline on 1xH100 (200 steps, for comparison)
- Date: 2026-04-07
- Hypothesis: Baseline at 200 steps for fair JEPA comparison
- Change: JEPA_ENABLED=0
- Hardware: 1xH100 SXM
- Result: val_bpb=2.7656 | ms/step=933 | artifact=4,982,207 bytes
- Post-EMA val_bpb=3.3702 | final_int6 val_bpb=3.4144
- Verdict: JEPA neutral at 200 steps (+0.0015 BPB). Expected — regularizers need thousands of steps.
- Cost: ~$0.50

## Experiment 1c: JEPA on 8xH100 (INTERRUPTED)
- Date: 2026-04-07
- Hypothesis: JEPA improves BPB over full 600s training on 8xH100
- Change: JEPA_ENABLED=1, full 600s wallclock
- Hardware: 8xH100 SXM ($21.52/hr)
- Result: **INTERRUPTED** — pod terminated at step 10/6400
- Measured: 93ms/step (SOTA is 86.7ms — only 6ms JEPA overhead)
- train_loss at step 10: 6.0584 (loss dropping normally)
- Verdict: CODE WORKS, need to rerun. Waiting for compute grant.
- Cost: ~$7 (pod ran ~20 min including setup + compile + 10 steps)

---

## Research Notes (no experiment needed)

### 8192 Vocab Analysis (2026-03-28)
- INVESTIGATED: Can we fit 8192 vocab in the SOTA's 16MB budget?
- ANSWER: No. SOTA artifact has only 9,994 bytes headroom. Even factored 8192×64 + 64×512 needs +25KB.
- Only viable with ternary quantization (4x more compact) which is a completely different codebase.
- DECISION: Stay on int6/GPTQ path. Don't pursue 8192 vocab.

### Depth Recurrence (2026-03-28, REVISED 2026-04-07)
- Was CONFIRMED DEAD END by 3 teams (PR #363) as of March 2026
- **NOW WORKING** — PR #1394 and #1435 use depth recurrence successfully
- Key: 2-3 layer loops on specific layers (4-5), not full recurrence
- Combined with SP8192 vocab in PR #1394 (1.086 BPB)
- Our dead-ends list needs updating — depth recurrence is viable again

### Novel Architectures (2026-03-28, PR #831)
- 6 architectures tested, all failed at 16MB/600s constraint
- SOTA stack is co-optimized: Parallel Muon + torch.compile + int6 + H100 tensor cores
- Throughput tax: must improve BPB by 0.007 per ms/step overhead
- SSM hybrid (Mamba) is 3.4x slower — completely unviable
