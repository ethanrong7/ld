# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath` | 0.773 | 0.826 | 0.831 | 24.9 | 0.23 | 10 |
| 2 | `field_coh` | 0.767 | 0.817 | 0.813 | 25.5 | 0.23 | 10 |
| 3 | `field_lag` | 0.731 | 0.778 | 0.773 | 27.0 | 0.21 | 10 |
| 4 | `field` | 0.730 | 0.782 | 0.777 | 26.0 | 0.21 | 10 |
| 5 | `accum` | 0.691 | 0.765 | 0.725 | 27.3 | 0.20 | 10 |
| 6 | `paper_outlier_rank` | 0.634 | 0.706 | 0.679 | 31.5 | 0.18 | 10 |
| 7 | `chain` | 0.573 | 0.639 | 0.602 | 44.6 | 0.27 | 10 |
| 8 | `paper` | 0.455 | 0.511 | 0.478 | 126.6 | 0.24 | 10 |
| 9 | `paper_outlier` | 0.455 | 0.511 | 0.478 | 126.6 | 0.24 | 10 |
| 10 | `outlier` | 0.351 | 0.413 | 0.359 | 113.3 | 0.20 | 10 |

## Per-clip `within_r` — leader (`fpath`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.512 | 55.5 | 0.806 | 141 |
| t10 | 0.804 | 21.4 | 0.928 | 151 |
| t2 | 0.928 | 16.0 | 0.945 | 471 |
| t3 | 0.721 | 27.1 | 0.871 | 95 |
| t4 | 0.722 | 24.5 | 0.952 | 106 |
| t5 | 0.621 | 29.1 | 0.933 | 130 |
| t6 | 0.853 | 25.1 | 0.929 | 657 |
| t7 | 0.998 | 14.9 | 1.000 | never |
| t8 | 0.644 | 30.8 | 0.917 | 228 |
| t9 | 0.926 | 24.7 | 0.968 | 238 |
