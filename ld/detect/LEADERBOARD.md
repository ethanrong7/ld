# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: t1, t2, t3, t4, t5, t6, t7, t8, t9, t10

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_human` | 0.940 | 0.965 | 0.950 | 20.8 | 0.58 | 10 |
| 2 | `fpath` | 0.797 | 0.841 | 0.829 | 23.6 | 0.35 | 10 |
| — | `oracle` (ceiling) | 0.958 | — | — | — | — | 10 |

## Per-clip `within_r` — leader (`fpath_human`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| t1 | 0.909 | 25.0 | 0.952 | 136 |
| t2 | 0.995 | 17.1 | 0.978 | never |
| t3 | 0.929 | 24.9 | 0.918 | 177 |
| t4 | 0.883 | 19.5 | 0.944 | 427 |
| t5 | 0.860 | 21.0 | 0.948 | 224 |
| t6 | 0.972 | 20.5 | 0.970 | never |
| t7 | 1.000 | 14.1 | 0.998 | never |
| t8 | 0.911 | 24.2 | 0.939 | 244 |
| t9 | 0.961 | 19.7 | 0.973 | 476 |
| t10 | 0.983 | 23.0 | 0.964 | never |
| **mean** | **0.940** | **20.9** | **0.958** | n/a |
