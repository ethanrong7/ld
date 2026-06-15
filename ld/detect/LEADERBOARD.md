# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `field_lag` | 0.721 | 0.766 | 0.755 | 27.5 | 0.16 | 10 |
| 2 | `field` | 0.714 | 0.768 | 0.752 | 27.0 | 0.15 | 10 |
| 3 | `fpath` | 0.698 | 0.745 | 0.741 | 28.1 | 0.21 | 10 |
| 4 | `accum` | 0.486 | 0.539 | 0.507 | 83.0 | 0.10 | 10 |
| 5 | `paper_outlier_rank` | 0.347 | 0.408 | 0.370 | 134.8 | 0.14 | 10 |
| 6 | `paper` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 7 | `paper_outlier` | 0.298 | 0.341 | 0.313 | 201.1 | 0.21 | 10 |
| 8 | `chain` | 0.283 | 0.328 | 0.297 | 201.1 | 0.20 | 10 |
| 9 | `outlier` | 0.187 | 0.256 | 0.186 | 188.7 | 0.03 | 10 |

## Per-clip `within_r` — leader (`field_lag`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.816 | 26.8 | 0.890 | 143 |
| t10 | 0.715 | 29.1 | 0.986 | 157 |
| t2 | 0.888 | 20.1 | 0.978 | 465 |
| t3 | 0.709 | 28.3 | 0.889 | 178 |
| t4 | 0.718 | 25.9 | 0.916 | 103 |
| t5 | 0.409 | 97.0 | 0.836 | 131 |
| t6 | 0.814 | 26.0 | 0.952 | 648 |
| t7 | 0.796 | 24.0 | 0.996 | 451 |
| t8 | 0.585 | 40.0 | 0.938 | 230 |
| t9 | 0.763 | 31.2 | 0.913 | 235 |
