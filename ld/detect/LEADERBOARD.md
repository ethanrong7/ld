# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `field` | 0.714 | 0.768 | 0.752 | 27.0 | 0.15 | 10 |
| 2 | `accum` | 0.486 | 0.539 | 0.507 | 83.0 | 0.10 | 10 |
| 3 | `paper_outlier_rank` | 0.347 | 0.408 | 0.370 | 134.8 | 0.14 | 10 |
| 4 | `paper` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 5 | `paper_outlier` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 6 | `chain` | 0.283 | 0.328 | 0.297 | 201.1 | 0.20 | 10 |
| 7 | `outlier` | 0.187 | 0.256 | 0.186 | 188.7 | 0.03 | 10 |

## Per-clip `within_r` — leader (`field`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.804 | 25.5 | 0.890 | 146 |
| t10 | 0.698 | 29.7 | 0.986 | 157 |
| t2 | 0.873 | 18.2 | 0.978 | 468 |
| t3 | 0.692 | 27.5 | 0.889 | 94 |
| t4 | 0.715 | 26.0 | 0.916 | 103 |
| t5 | 0.409 | 99.9 | 0.836 | 133 |
| t6 | 0.819 | 26.5 | 0.952 | 648 |
| t7 | 0.796 | 22.9 | 0.996 | 451 |
| t8 | 0.589 | 44.9 | 0.938 | 230 |
| t9 | 0.751 | 31.2 | 0.913 | 238 |
