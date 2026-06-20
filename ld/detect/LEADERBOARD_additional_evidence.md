# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` · conf=0.25 · imgsz=768 · clips: a01_ld1080p1, a02_ld1080p2, a04_maplestory_2026_04_19_1_12_53_pm, a05_maplestory_2026_04_19_2_09_05_pm, a06_maplestory_2026_05_24_3_04_32_pm, a08_maplestory_2026_05_24_5_15_42_pm, a09_maplestory_2026_05_24_5_51_39_pm, a10_maplestory_2026_05_24_9_04_20_pm, a11_maplestory_2026_05_25_1_24_55_pm, a12_maplestory_2026_05_25_10_42_47_pm, a13_maplestory_2026_05_25_2_14_31_pm, a14_maplestory_2026_05_25_7_42_36_pm, a15_maplestory_2026_05_25_8_07_11_pm

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_freeze` | 0.837 | 0.928 | 0.894 | 25.0 | 0.22 | 13 |
| 2 | `fpath_fuse` | 0.814 | 0.911 | 0.888 | 27.8 | 0.32 | 13 |
| 3 | `fpath_hedge` | 0.811 | 0.906 | 0.873 | 26.7 | 0.26 | 13 |
| 4 | `fpath_hyst` | 0.800 | 0.895 | 0.874 | 27.8 | 0.32 | 13 |
| 5 | `fpath_coh` | 0.800 | 0.895 | 0.873 | 29.2 | 0.32 | 13 |
| 6 | `fpath` | 0.780 | 0.876 | 0.850 | 30.9 | 0.25 | 13 |
| 7 | `field_coh` | 0.741 | 0.850 | 0.796 | 31.2 | 0.26 | 13 |
| 8 | `field` | 0.740 | 0.837 | 0.804 | 33.5 | 0.25 | 13 |
| 9 | `field_lag` | 0.730 | 0.839 | 0.783 | 31.3 | 0.23 | 13 |

## Per-clip `within_r` — leader (`fpath_freeze`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| a01_ld1080p1 | 0.841 | 24.6 | 0.856 | 562 |
| a02_ld1080p2 | 0.952 | 25.0 | 0.977 | 752 |
| a04_maplestory_2026_04_19_1_12_53_pm | 0.871 | 31.1 | 0.914 | 226 |
| a05_maplestory_2026_04_19_2_09_05_pm | 0.979 | 20.2 | 0.997 | 963 |
| a06_maplestory_2026_05_24_3_04_32_pm | 0.691 | 31.5 | 0.937 | 170 |
| a08_maplestory_2026_05_24_5_15_42_pm | 0.769 | 22.7 | 0.828 | 443 |
| a09_maplestory_2026_05_24_5_51_39_pm | 0.767 | 23.8 | 0.782 | 387 |
| a10_maplestory_2026_05_24_9_04_20_pm | 0.828 | 29.0 | 0.909 | 165 |
| a11_maplestory_2026_05_25_1_24_55_pm | 0.916 | 17.9 | 0.985 | 501 |
| a12_maplestory_2026_05_25_10_42_47_pm | 0.762 | 32.1 | 0.945 | 272 |
| a13_maplestory_2026_05_25_2_14_31_pm | 0.821 | 29.3 | 0.886 | 164 |
| a14_maplestory_2026_05_25_7_42_36_pm | 0.852 | 24.5 | 0.966 | 140 |
| a15_maplestory_2026_05_25_8_07_11_pm | 0.830 | 33.7 | 0.929 | 161 |
| **mean** | **0.837** | **26.6** | **0.916** | n/a |
