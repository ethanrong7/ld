# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` � conf=0.25 � imgsz=768 � clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_freeze` | 0.932 | 0.961 | 0.948 | 22.1 | 0.54 | 10 |
| 2 | `fpath_hedge` | 0.899 | 0.940 | 0.922 | 22.7 | 0.52 | 10 |
| 3 | `fpath_hyst` | 0.878 | 0.928 | 0.915 | 23.1 | 0.46 | 10 |
| 4 | `fpath_fuse` | 0.876 | 0.927 | 0.913 | 23.1 | 0.46 | 10 |
| 5 | `fpath_coh` | 0.825 | 0.869 | 0.860 | 23.4 | 0.41 | 10 |
| 6 | `fpath` | 0.797 | 0.841 | 0.829 | 23.6 | 0.35 | 10 |
| 7 | `field_coh` | 0.773 | 0.810 | 0.794 | 25.9 | 0.24 | 10 |
| 8 | `field_lag` | 0.749 | 0.785 | 0.769 | 26.4 | 0.23 | 10 |
| 9 | `field` | 0.744 | 0.785 | 0.769 | 25.3 | 0.22 | 10 |

## Per-clip `within_r` � leader (`fpath_freeze`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.895 | 22.9 | 0.952 | 135 |
| t2 | 0.978 | 15.2 | 0.978 | never |
| t3 | 0.913 | 25.2 | 0.918 | 175 |
| t4 | 0.880 | 19.3 | 0.944 | 426 |
| t5 | 0.858 | 21.9 | 0.948 | 223 |
| t6 | 0.959 | 20.7 | 0.970 | 672 |
| t7 | 1.000 | 17.3 | 0.998 | never |
| t8 | 0.909 | 26.5 | 0.939 | 242 |
| t9 | 0.958 | 22.2 | 0.973 | 474 |
| t10 | 0.974 | 22.4 | 0.964 | never |
| **mean** | **0.932** | **21.4** | **0.958** | n/a |
