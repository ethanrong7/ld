# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_coh` | 0.825 | 0.869 | 0.860 | 23.4 | 0.41 | 10 |
| 2 | `fpath` | 0.797 | 0.841 | 0.829 | 23.6 | 0.35 | 10 |
| 3 | `fpath_reacq` | 0.783 | 0.827 | 0.815 | 24.0 | 0.35 | 10 |
| 4 | `field_coh` | 0.773 | 0.810 | 0.794 | 25.9 | 0.24 | 10 |
| 5 | `field_lag` | 0.749 | 0.785 | 0.769 | 26.4 | 0.23 | 10 |
| 6 | `field` | 0.744 | 0.785 | 0.769 | 25.3 | 0.22 | 10 |
| 7 | `paper` | 0.618 | 0.680 | 0.638 | 32.6 | 0.42 | 10 |
| 8 | `paper_outlier` | 0.618 | 0.680 | 0.638 | 32.6 | 0.42 | 10 |
| 9 | `paper_outlier_rank` | 0.616 | 0.693 | 0.636 | 36.7 | 0.11 | 10 |
| 10 | `accum` | 0.590 | 0.677 | 0.604 | 49.2 | 0.19 | 10 |
| 11 | `chain` | 0.585 | 0.649 | 0.604 | 33.1 | 0.34 | 10 |
| 12 | `outlier` | 0.278 | 0.355 | 0.279 | 167.6 | 0.03 | 10 |

## Per-clip `within_r` — leader (`fpath_coh`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.702 | 28.3 | 0.952 | 142 |
| t10 | 0.915 | 21.9 | 0.964 | 150 |
| t2 | 0.955 | 15.6 | 0.978 | never |
| t3 | 0.810 | 25.7 | 0.918 | 160 |
| t4 | 0.745 | 20.5 | 0.944 | 389 |
| t5 | 0.617 | 27.7 | 0.948 | 138 |
| t6 | 0.877 | 24.2 | 0.970 | 662 |
| t7 | 0.991 | 17.5 | 0.998 | never |
| t8 | 0.732 | 33.4 | 0.939 | 239 |
| t9 | 0.909 | 22.7 | 0.973 | 468 |
