# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_fuse` | 0.876 | 0.927 | 0.913 | 23.1 | 0.46 | 10 |
| 2 | `fpath_coh` | 0.825 | 0.869 | 0.860 | 23.4 | 0.41 | 10 |
| 3 | `fpath` | 0.797 | 0.841 | 0.829 | 23.6 | 0.35 | 10 |
| 4 | `fpath_reacq` | 0.783 | 0.827 | 0.815 | 24.0 | 0.35 | 10 |
| 5 | `field_coh` | 0.773 | 0.810 | 0.794 | 25.9 | 0.24 | 10 |
| 6 | `field_lag` | 0.749 | 0.785 | 0.769 | 26.4 | 0.23 | 10 |
| 7 | `field` | 0.744 | 0.785 | 0.769 | 25.3 | 0.22 | 10 |
| 8 | `paper` | 0.618 | 0.680 | 0.638 | 32.6 | 0.42 | 10 |
| 9 | `paper_outlier` | 0.618 | 0.680 | 0.638 | 32.6 | 0.42 | 10 |
| 10 | `paper_outlier_rank` | 0.616 | 0.693 | 0.636 | 36.7 | 0.11 | 10 |
| 11 | `accum` | 0.590 | 0.677 | 0.604 | 49.2 | 0.19 | 10 |
| 12 | `chain` | 0.585 | 0.649 | 0.604 | 33.1 | 0.34 | 10 |
| 13 | `outlier` | 0.278 | 0.355 | 0.279 | 167.6 | 0.03 | 10 |

## Per-clip `within_r` — leader (`fpath_fuse`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.783 | 25.0 | 0.952 | 133 |
| t10 | 0.937 | 21.6 | 0.964 | 325 |
| t2 | 0.978 | 15.3 | 0.978 | never |
| t3 | 0.852 | 24.8 | 0.918 | 160 |
| t4 | 0.849 | 20.0 | 0.944 | 389 |
| t5 | 0.759 | 25.6 | 0.948 | 206 |
| t6 | 0.922 | 23.4 | 0.970 | 672 |
| t7 | 0.989 | 17.8 | 0.998 | never |
| t8 | 0.767 | 30.0 | 0.939 | 239 |
| t9 | 0.924 | 22.8 | 0.973 | 472 |
