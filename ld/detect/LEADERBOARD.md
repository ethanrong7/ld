# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `paper_outlier_rank` | 0.347 | 0.408 | 0.370 | 134.8 | 0.14 | 10 |
| 2 | `trajectory_paper` | 0.336 | 0.417 | 0.356 | 120.2 | 0.03 | 10 |
| 3 | `trajectory` | 0.334 | 0.420 | 0.354 | 119.6 | 0.03 | 10 |
| 4 | `hypothesis_paper` | 0.312 | 0.377 | 0.321 | 159.7 | 0.04 | 10 |
| 5 | `hybrid_unified` | 0.307 | 0.350 | 0.321 | 201.1 | 0.20 | 10 |
| 6 | `paper` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 7 | `trajectory_reacq` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 8 | `paper_outlier` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 9 | `hypothesis` | 0.285 | 0.355 | 0.292 | 158.2 | 0.01 | 10 |
| 10 | `chain` | 0.283 | 0.328 | 0.297 | 201.1 | 0.20 | 10 |
| 11 | `hybrid` | 0.253 | 0.292 | 0.265 | 212.9 | 0.20 | 10 |
| 12 | `outlier` | 0.187 | 0.256 | 0.186 | 188.7 | 0.03 | 10 |
| 13 | `paper_reacq` | 0.142 | 0.227 | 0.149 | 186.9 | 0.02 | 10 |
| 14 | `paper_free` | 0.138 | 0.227 | 0.144 | 178.8 | 0.02 | 10 |

## Per-clip `within_r` — leader (`paper_outlier_rank`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.504 | 57.5 | 0.890 | 116 |
| t10 | 0.321 | 139.8 | 0.986 | 201 |
| t2 | 0.132 | 308.1 | 0.978 | 404 |
| t3 | 0.272 | 171.0 | 0.889 | 134 |
| t4 | 0.217 | 196.5 | 0.916 | 103 |
| t5 | 0.305 | 129.9 | 0.836 | 203 |
| t6 | 0.391 | 116.4 | 0.952 | 432 |
| t7 | 0.633 | 34.4 | 0.996 | 701 |
| t8 | 0.352 | 146.9 | 0.938 | 148 |
| t9 | 0.341 | 124.2 | 0.913 | 205 |
