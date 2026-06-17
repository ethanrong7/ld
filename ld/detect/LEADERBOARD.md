# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath` | 0.855 | 0.891 | 0.877 | 22.4 | 0.33 | 10 |
| 2 | `fpath_reacq` | 0.845 | 0.881 | 0.867 | 22.4 | 0.33 | 10 |
| 3 | `field_coh` | 0.773 | 0.803 | 0.789 | 25.0 | 0.22 | 10 |
| 4 | `field_lag` | 0.746 | 0.776 | 0.761 | 25.6 | 0.26 | 10 |
| 5 | `field` | 0.741 | 0.777 | 0.760 | 26.1 | 0.28 | 10 |
| 6 | `accum` | 0.706 | 0.782 | 0.717 | 29.7 | 0.20 | 10 |
| 7 | `paper_outlier_rank` | 0.652 | 0.728 | 0.664 | 33.5 | 0.12 | 10 |
| 8 | `paper` | 0.625 | 0.696 | 0.637 | 35.6 | 0.28 | 10 |
| 9 | `paper_outlier` | 0.625 | 0.696 | 0.637 | 35.6 | 0.28 | 10 |
| 10 | `chain` | 0.454 | 0.513 | 0.462 | 133.7 | 0.28 | 10 |
| 11 | `outlier` | 0.376 | 0.438 | 0.377 | 119.2 | 0.12 | 10 |

## Per-clip `within_r` — leader (`fpath`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.806 | 22.6 | 0.983 | 146 |
| t10 | 0.957 | 20.2 | 0.990 | 152 |
| t2 | 0.963 | 15.7 | 0.975 | 520 |
| t3 | 0.902 | 25.1 | 0.943 | 282 |
| t4 | 0.729 | 22.3 | 0.953 | 109 |
| t5 | 0.718 | 25.5 | 0.974 | 212 |
| t6 | 0.875 | 24.1 | 0.989 | 663 |
| t7 | 0.993 | 16.5 | 1.000 | never |
| t8 | 0.664 | 33.7 | 0.971 | 232 |
| t9 | 0.945 | 21.8 | 0.966 | 474 |
