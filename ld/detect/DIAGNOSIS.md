# Identity-loss decomposition

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

Each rung relaxes one constraint; the gap between rungs assigns the loss. R0 detection ceiling · R1 best single tracklet (association) · R2 GT box is top motion outlier per frame (signal) · accum = live tracker actual. `persist` = fraction of oracle-hit frames covered by one tracklet identity (low ⇒ identity fragments).

| clip | R0 det | R1 assoc(IoU) | R1 assoc(MC) | persist IoU/MC | R2 resid top1/3 | R2 fused top1/3 | accum |
|------|-------:|--------------:|-------------:|---------------|-----------------|-----------------|------:|
| t1 | 0.89 | 0.18 | 0.28 | 0.18/0.16 | 0.14/0.35 | 0.13/0.36 | 0.31 |
| t2 | 0.98 | 0.41 | 0.41 | 0.42/0.42 | 0.13/0.46 | 0.13/0.45 | 0.90 |
| t3 | 0.89 | 0.22 | 0.22 | 0.24/0.19 | 0.14/0.39 | 0.14/0.38 | 0.56 |
| t4 | 0.92 | 0.20 | 0.19 | 0.15/0.15 | 0.14/0.38 | 0.14/0.37 | 0.33 |
| t5 | 0.84 | 0.24 | 0.24 | 0.25/0.26 | 0.17/0.45 | 0.17/0.45 | 0.70 |
| t6 | 0.95 | 0.17 | 0.15 | 0.13/0.11 | 0.17/0.41 | 0.17/0.40 | 0.50 |
| t7 | 1.00 | 0.19 | 0.18 | 0.14/0.15 | 0.13/0.38 | 0.12/0.38 | 0.22 |
| t8 | 0.94 | 0.29 | 0.21 | 0.27/0.20 | 0.11/0.30 | 0.11/0.28 | 0.55 |
| t9 | 0.91 | 0.21 | 0.18 | 0.17/0.17 | 0.13/0.36 | 0.13/0.36 | 0.39 |
| t10 | 0.99 | 0.28 | 0.24 | 0.22/0.22 | 0.11/0.30 | 0.10/0.29 | 0.40 |
| **mean** | **0.93** | **0.24** | **0.23** | 0.22/0.20 | 0.14/0.38 | 0.13/0.37 | **0.49** |

## Reading

- **Detection is fine** (R0 0.93); the loss is downstream.
- **Identity fragments badly**: a single tracklet holds the real shape only 0.24 of the time (persistence 0.22); motion-compensated association does not help (0.23). The tracklet abstraction is structurally leaky for an independently-moving target amid dense identical neighbours.
- **accum (0.49) already EXCEEDS the single-tracklet ceiling (0.24)** — its re-bind/switch logic stitches fragments back together. So the remaining loss is NOT decision-tuning headroom (accum beats the naive single-identity bound by ~2x).
- **Per-frame signal is weak**: the GT box is the top fused motion outlier only 0.13 of frames (top-3 0.37) — the real shape is often stationary or buried, so a memoryless picker is hopeless and temporal integration is essential.

**Conclusion — the binding limiters are association fragmentation + weak per-frame signal, not decision thresholds.** Highest-EV next moves: (1) a **position-field accumulator** that integrates independent-motion evidence *spatially* (reuse `ld/vision/motion.py` `saliency_map`), immune to identity fragmentation; (2) **stronger per-frame signal**. Decision tuning (adaptive switch/tie-break) is low-EV — accum already transcends the single-identity ceiling.
