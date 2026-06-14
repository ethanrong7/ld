# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `accum` | 0.486 | 0.539 | 0.507 | 83.0 | 0.10 | 10 |
| 2 | `paper_outlier_rank` | 0.347 | 0.408 | 0.370 | 134.8 | 0.14 | 10 |
| 3 | `trajectory_paper` | 0.336 | 0.417 | 0.356 | 120.2 | 0.03 | 10 |
| 4 | `trajectory` | 0.334 | 0.420 | 0.354 | 119.6 | 0.03 | 10 |
| 5 | `hypothesis_paper` | 0.312 | 0.377 | 0.321 | 159.7 | 0.04 | 10 |
| 6 | `hybrid_unified` | 0.307 | 0.350 | 0.321 | 201.1 | 0.20 | 10 |
| 7 | `paper` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 8 | `trajectory_reacq` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 9 | `paper_outlier` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 10 | `hypothesis` | 0.285 | 0.355 | 0.292 | 158.2 | 0.01 | 10 |
| 11 | `chain` | 0.283 | 0.328 | 0.297 | 201.1 | 0.20 | 10 |
| 12 | `hybrid` | 0.253 | 0.292 | 0.265 | 212.9 | 0.20 | 10 |
| 13 | `outlier` | 0.187 | 0.256 | 0.186 | 188.7 | 0.03 | 10 |
| 14 | `paper_reacq` | 0.142 | 0.227 | 0.149 | 186.9 | 0.02 | 10 |
| 15 | `paper_free` | 0.138 | 0.227 | 0.144 | 178.8 | 0.02 | 10 |

## Per-clip `within_r` — leader (`accum`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.308 | 135.7 | 0.890 | 116 |
| t10 | 0.399 | 142.8 | 0.986 | 152 |
| t2 | 0.903 | 20.2 | 0.978 | 398 |
| t3 | 0.557 | 31.2 | 0.889 | 95 |
| t4 | 0.334 | 110.7 | 0.916 | 103 |
| t5 | 0.701 | 28.6 | 0.836 | 296 |
| t6 | 0.501 | 52.5 | 0.952 | 510 |
| t7 | 0.220 | 221.7 | 0.996 | 446 |
| t8 | 0.550 | 55.3 | 0.938 | 192 |
| t9 | 0.387 | 133.6 | 0.913 | 142 |
