# Fixed-lag Viterbi ceiling (over the `field` signal)

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

Best a temporal integrator could do with a bounded lookahead of K frames, using the **exact** per-frame signal the `field` tracker consumes (independent-motion saliency mass per YOLO box). K=0 causal · K=15 ~0.5s lag (physically free) · K=inf full offline. `trans_w` chosen once per lag by best mean across clips (no per-clip tuning).

| clip | field (causal) | K=0 | K=5 | K=15 | K=30 | K=inf |
|------|---:|---:|---:|---:|---:|---:|
| t1 | 0.804 | 0.570 | 0.587 | 0.591 | 0.593 | 0.663 |
| t2 | 0.873 | 0.953 | 0.963 | 0.963 | 0.963 | 0.970 |
| t3 | 0.692 | 0.544 | 0.587 | 0.605 | 0.625 | 0.740 |
| t4 | 0.715 | 0.637 | 0.650 | 0.671 | 0.684 | 0.670 |
| t5 | 0.409 | 0.303 | 0.338 | 0.366 | 0.389 | 0.314 |
| t6 | 0.819 | 0.849 | 0.864 | 0.901 | 0.937 | 0.937 |
| t7 | 0.796 | 0.958 | 0.958 | 0.958 | 0.971 | 0.987 |
| t8 | 0.589 | 0.658 | 0.661 | 0.695 | 0.718 | 0.784 |
| t9 | 0.751 | 0.829 | 0.829 | 0.832 | 0.864 | 0.857 |
| t10 | 0.698 | 0.937 | 0.945 | 0.952 | 0.952 | 0.957 |
| **mean** | **0.715** | **0.724** | **0.738** | **0.753** | **0.770** | **0.788** |

trans_w per K: K=0:1.0, K=5:1.0, K=15:1.0, K=30:1.0, K=inf:2.0

## Reading

- Causal ceiling (K=0) over this signal = **0.724**; `field` actual = **0.715**.
- Lookahead gain: K=0→15 = **+0.030**, K=0→inf = **+0.064**.
- **INTEGRATION-LIMITED (lookahead helps).** A ~0.5s fixed-lag smoother gains +0.030 and is legitimately online (hold K frames, emit t-K). Worth building.
- Note: `field` (0.715) ~matches/exceeds the K=0 Viterbi ceiling — the causal greedy is already near-optimal on this signal, reinforcing that the signal (not the decision) is the limiter.
