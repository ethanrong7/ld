# Identity mode leaderboard

Weights: `data/detect/runs/yolov8n_single_combined/weights/best.pt` � conf=0.25 � imgsz=768 � clips: a01_ld1080p1, a02_ld1080p2, a04_maplestory_2026_04_19_1_12_53_pm, a05_maplestory_2026_04_19_2_09_05_pm, a06_maplestory_2026_05_24_3_04_32_pm, a08_maplestory_2026_05_24_5_15_42_pm, a09_maplestory_2026_05_24_5_51_39_pm, a10_maplestory_2026_05_24_9_04_20_pm, a11_maplestory_2026_05_25_1_24_55_pm, a12_maplestory_2026_05_25_10_42_47_pm, a13_maplestory_2026_05_25_2_14_31_pm, a14_maplestory_2026_05_25_7_42_36_pm, a15_maplestory_2026_05_25_8_07_11_pm

`within_r` = fraction of scored frames the estimate lands inside the shape radius of GT. `drift_frac` = mean fraction of the scored span before a sustained drift begins (1.0 = never drifts). Ranked by mean `within_r`.

| rank | mode | within_r | within_1.5r | conditional | median_px | drift_frac | clips |
|-----:|------|---------:|------------:|------------:|----------:|-----------:|------:|
| 1 | `fpath_human` | 0.859 | 0.936 | 0.897 | 24.8 | 0.26 | 13 |
| 2 | `fpath_freeze` | 0.837 | 0.928 | 0.894 | 25.0 | 0.22 | 13 |
| 3 | `fpath_fuse` | 0.814 | 0.911 | 0.888 | 27.8 | 0.32 | 13 |
| 4 | `fpath_hedge` | 0.811 | 0.906 | 0.873 | 26.7 | 0.26 | 13 |
| 5 | `fpath_hyst` | 0.800 | 0.895 | 0.874 | 27.8 | 0.32 | 13 |
| 6 | `fpath_coh` | 0.800 | 0.895 | 0.873 | 29.2 | 0.32 | 13 |
| 7 | `fpath` | 0.780 | 0.876 | 0.850 | 30.9 | 0.25 | 13 |
| 8 | `field_coh` | 0.741 | 0.850 | 0.796 | 31.2 | 0.26 | 13 |
| 9 | `field` | 0.740 | 0.837 | 0.804 | 33.5 | 0.25 | 13 |
| 10 | `field_lag` | 0.730 | 0.839 | 0.783 | 31.3 | 0.23 | 13 |

`fpath_human` = `fpath_freeze` + human-cursor output dynamics (1-Euro + 2px deadband on the emitted point;
box decisions byte-identical). It tops the held-out board too: +0.022 over `fpath_freeze`, flat-or-up on every
clip (worst −0.003, a single-frame flip), reproducing the t1–t10 lift. The other modes are unchanged from the
prior regen (their tracking is identical).

## Per-clip `within_r` � leader (`fpath_human`)

| clip | within_r | median_px | oracle | drift_onset |
|------|---------:|----------:|-------:|------------:|
| a01_ld1080p1 | 0.890 | 21.8 | 0.856 | 564 |
| a02_ld1080p2 | 0.949 | 24.8 | 0.977 | 754 |
| a04_maplestory_2026_04_19_1_12_53_pm | 0.868 | 29.2 | 0.914 | 223 |
| a05_maplestory_2026_04_19_2_09_05_pm | 0.981 | 17.3 | 0.997 | 964 |
| a06_maplestory_2026_05_24_3_04_32_pm | 0.706 | 31.0 | 0.937 | 293 |
| a08_maplestory_2026_05_24_5_15_42_pm | 0.831 | 18.1 | 0.828 | 515 |
| a09_maplestory_2026_05_24_5_51_39_pm | 0.817 | 21.1 | 0.782 | 414 |
| a10_maplestory_2026_05_24_9_04_20_pm | 0.845 | 29.7 | 0.909 | 166 |
| a11_maplestory_2026_05_25_1_24_55_pm | 0.931 | 15.5 | 0.985 | 615 |
| a12_maplestory_2026_05_25_10_42_47_pm | 0.766 | 29.7 | 0.945 | 271 |
| a13_maplestory_2026_05_25_2_14_31_pm | 0.841 | 30.3 | 0.886 | 167 |
| a14_maplestory_2026_05_25_7_42_36_pm | 0.877 | 22.1 | 0.966 | 140 |
| a15_maplestory_2026_05_25_8_07_11_pm | 0.864 | 31.1 | 0.929 | 167 |
| **mean** | **0.859** | **24.8** | **0.916** | n/a |
