
### Main results (clean test set)

| dataset | run | model | runs | acc | macro_f1 | weighted_f1 | hate_f1 | ece | brier |
|---|---|---|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1 | 93.9 | 61.4 | 92.7 | 25.9 | 5.2 | 5.7 |
| dv | dv_baseline | teacher | 1 | 94.2 | 48.5 | 91.4 | 0.0 | 0.5 | 5.4 |
| dv | dv_ours_emoji | student | 1 | 89.5 | 67.5 | 91.2 | 40.8 | 23.1 | 13.6 |
| dv | dv_ours_emoji | teacher | 1 | 90.8 | 68.3 | 92.0 | 41.6 | 20.8 | 11.4 |
| dv | dv_ours_gate | student | 1 | 92.3 | 69.5 | 92.8 | 43.1 | 20.5 | 10.3 |
| dv | dv_ours_gate_cb | student | 1 | 90.2 | 68.5 | 91.6 | 42.4 | 21.4 | 12.2 |
| dv | dv_ours_hd | student | 1 | 91.7 | 69.9 | 92.6 | 44.2 | 26.8 | 13.8 |
| dv | dv_ours_hd | teacher | 1 | 92.2 | 66.1 | 92.4 | 36.3 | 26.1 | 11.7 |
| dv | dv_rkd | student | 1 | 92.9 | 69.6 | 93.1 | 43.0 | 24.0 | 11.2 |
| dv | dv_rkd | teacher | 1 | 91.5 | 66.7 | 92.1 | 38.1 | 21.0 | 10.2 |
| se | se_abl_emo_no_cb | student | 1 | 59.9 | 59.8 | 59.5 | 61.8 | 3.5 | 23.7 |
| se | se_abl_emo_no_gate | student | 1 | 57.2 | 56.6 | 55.8 | 61.8 | 3.5 | 24.1 |
| se | se_abl_emo_no_revise | student | 1 | 58.8 | 58.5 | 58.0 | 61.8 | 2.8 | 23.8 |
| se | se_abl_emoji_paperkd | student | 1 | 57.3 | 56.9 | 56.3 | 60.9 | 4.7 | 23.8 |
| se | se_abl_emoji_paperkd | teacher | 1 | 59.8 | 59.7 | 59.4 | 61.9 | 17.4 | 27.0 |
| se | se_abl_kd_only | student | 1 | 57.1 | 56.3 | 55.4 | 62.3 | 10.3 | 24.9 |
| se | se_abl_spn_paperkd | student | 1 | 51.8 | 49.6 | 47.9 | 60.2 | 7.2 | 25.6 |
| se | se_abl_spnhd_paperkd | student | 1 | 49.2 | 45.7 | 43.5 | 59.5 | 14.4 | 27.3 |
| se | se_baseline | student | 3 | 49.7 ± 2.7 | 44.9 ± 4.6 | 42.4 ± 5.3 | 60.8 ± 0.6 | 34.9 ± 2.4 | 38.3 ± 1.8 |
| se | se_baseline | teacher | 3 | 46.3 ± 0.6 | 39.3 ± 1.2 | 36.1 ± 1.4 | 59.8 ± 0.1 | 39.8 ± 4.1 | 41.7 ± 3.1 |
| se | se_ours | student | 1 | 54.8 | 53.5 | 52.3 | 61.3 | 16.1 | 28.7 |
| se | se_ours | teacher | 1 | 55.5 | 54.7 | 53.7 | 60.8 | 22.5 | 30.8 |
| se | se_ours_emoji | student | 3 | 58.0 ± 1.1 | 57.6 ± 1.3 | 56.9 ± 1.4 | 61.7 ± 0.6 | 3.8 ± 0.4 | 23.8 ± 0.2 |
| se | se_ours_emoji | teacher | 2 | 60.5 ± 0.0 | 60.5 ± 0.0 | 60.4 ± 0.1 | 60.9 ± 0.2 | 7.0 ± 0.2 | 23.6 ± 0.0 |
| se | se_ours_gate | student | 3 | 60.9 ± 1.1 | 60.8 ± 1.1 | 60.7 ± 1.2 | 61.9 ± 0.5 | 3.9 ± 0.9 | 23.3 ± 0.3 |
| se | se_ours_hd | student | 3 | 48.9 ± 3.1 | 44.1 ± 5.5 | 41.6 ± 6.2 | 60.2 ± 0.5 | 13.5 ± 1.7 | 26.9 ± 0.1 |
| se | se_ours_hd | teacher | 3 | 53.6 ± 2.9 | 52.0 ± 4.0 | 50.7 ± 4.6 | 60.1 ± 0.5 | 15.4 ± 3.4 | 27.6 ± 1.1 |
| se | se_rkd | student | 3 | 56.6 ± 2.9 | 55.7 ± 3.8 | 54.7 ± 4.3 | 61.7 ± 0.6 | 5.6 ± 2.5 | 24.0 ± 0.6 |
| se | se_rkd | teacher | 3 | 58.9 ± 0.7 | 58.7 ± 0.7 | 58.3 ± 0.8 | 61.5 ± 0.6 | 14.1 ± 0.4 | 25.7 ± 0.4 |

### Robustness (perturbed test set)

| dataset | run | model | runs | acc | macro_f1 | weighted_f1 | hate_f1 |
|---|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1 | 92.0 | 60.6 | 91.7 | 25.4 |
| dv | dv_baseline | teacher | 1 | 94.2 | 48.5 | 91.4 | 0.0 |
| dv | dv_ours_emoji | student | 1 | 72.2 | 52.9 | 79.6 | 22.7 |
| dv | dv_ours_emoji | teacher | 1 | 75.9 | 54.2 | 82.1 | 22.8 |
| dv | dv_ours_gate | student | 1 | 80.3 | 56.9 | 85.0 | 25.1 |
| dv | dv_ours_gate_cb | student | 1 | 73.3 | 53.2 | 80.3 | 22.5 |
| dv | dv_ours_hd | student | 1 | 78.8 | 56.6 | 84.0 | 25.7 |
| dv | dv_ours_hd | teacher | 1 | 87.3 | 59.4 | 89.2 | 25.7 |
| dv | dv_rkd | student | 1 | 85.2 | 59.5 | 88.0 | 27.3 |
| dv | dv_rkd | teacher | 1 | 84.4 | 58.9 | 87.5 | 26.6 |
| se | se_abl_emo_no_cb | student | 1 | 55.3 | 54.8 | 54.0 | 59.6 |
| se | se_abl_emo_no_gate | student | 1 | 53.4 | 52.1 | 50.8 | 60.0 |
| se | se_abl_emo_no_revise | student | 1 | 54.5 | 53.8 | 52.8 | 59.7 |
| se | se_abl_emoji_paperkd | student | 1 | 55.1 | 54.4 | 53.5 | 60.1 |
| se | se_abl_emoji_paperkd | teacher | 1 | 56.1 | 55.9 | 55.4 | 58.8 |
| se | se_abl_kd_only | student | 1 | 54.6 | 53.2 | 51.9 | 61.4 |
| se | se_abl_spn_paperkd | student | 1 | 49.7 | 47.4 | 45.7 | 58.4 |
| se | se_abl_spnhd_paperkd | student | 1 | 48.6 | 44.7 | 42.4 | 59.3 |
| se | se_baseline | student | 3 | 50.5 ± 1.3 | 48.2 ± 2.1 | 46.5 ± 2.5 | 59.0 ± 1.0 |
| se | se_baseline | teacher | 3 | 48.4 ± 0.1 | 45.1 ± 0.2 | 43.0 ± 0.2 | 58.5 ± 0.0 |
| se | se_ours | student | 1 | 51.5 | 49.2 | 47.5 | 59.9 |
| se | se_ours | teacher | 1 | 53.3 | 52.7 | 51.8 | 58.2 |
| se | se_ours_emoji | student | 3 | 55.3 ± 1.0 | 54.5 ± 1.0 | 53.5 ± 1.0 | 60.8 ± 0.9 |
| se | se_ours_emoji | teacher | 2 | 58.4 ± 0.3 | 58.3 ± 0.3 | 58.1 ± 0.2 | 60.2 ± 0.4 |
| se | se_ours_gate | student | 3 | 57.2 ± 1.2 | 56.9 ± 1.3 | 56.3 ± 1.4 | 60.5 ± 0.7 |
| se | se_ours_hd | student | 3 | 48.4 ± 1.8 | 43.8 ± 3.1 | 41.3 ± 3.5 | 59.7 ± 0.4 |
| se | se_ours_hd | teacher | 3 | 52.0 ± 2.0 | 50.4 ± 2.8 | 49.0 ± 3.3 | 59.0 ± 0.3 |
| se | se_rkd | student | 3 | 54.5 ± 2.6 | 53.1 ± 3.5 | 51.9 ± 3.9 | 60.8 ± 0.6 |
| se | se_rkd | teacher | 3 | 57.3 ± 1.3 | 56.9 ± 1.4 | 56.3 ± 1.5 | 60.7 ± 0.9 |

### Slice: tweets WITH emojis/hashtags

| dataset | run | model | runs | acc | macro_f1 | weighted_f1 | hate_f1 |
|---|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1 | 96.0 | 63.0 | 95.2 | 28.1 |
| dv | dv_baseline | teacher | 1 | 96.0 | 49.0 | 94.0 | 0.0 |
| dv | dv_ours_emoji | student | 1 | 92.3 | 65.0 | 93.4 | 34.1 |
| dv | dv_ours_emoji | teacher | 1 | 93.1 | 65.5 | 93.9 | 34.7 |
| dv | dv_ours_gate | student | 1 | 94.8 | 68.0 | 94.9 | 38.8 |
| dv | dv_ours_gate_cb | student | 1 | 93.0 | 67.4 | 94.0 | 38.5 |
| dv | dv_ours_hd | student | 1 | 93.9 | 67.2 | 94.4 | 37.5 |
| dv | dv_ours_hd | teacher | 1 | 94.3 | 61.8 | 94.2 | 26.7 |
| dv | dv_rkd | student | 1 | 94.7 | 62.7 | 94.5 | 28.2 |
| dv | dv_rkd | teacher | 1 | 92.3 | 59.4 | 93.0 | 22.8 |
| se | se_abl_emo_no_cb | student | 1 | 63.5 | 63.4 | 63.8 | 61.1 |
| se | se_abl_emo_no_gate | student | 1 | 61.6 | 61.6 | 61.5 | 61.9 |
| se | se_abl_emo_no_revise | student | 1 | 62.8 | 62.8 | 63.0 | 61.2 |
| se | se_abl_emoji_paperkd | student | 1 | 61.5 | 61.5 | 61.6 | 60.6 |
| se | se_abl_emoji_paperkd | teacher | 1 | 60.5 | 60.5 | 60.3 | 61.4 |
| se | se_abl_kd_only | student | 1 | 57.8 | 57.4 | 56.8 | 61.5 |
| se | se_abl_spn_paperkd | student | 1 | 47.6 | 44.1 | 41.9 | 58.0 |
| se | se_abl_spnhd_paperkd | student | 1 | 46.0 | 41.3 | 38.7 | 57.8 |
| se | se_baseline | student | 3 | 48.3 ± 1.6 | 43.5 ± 2.8 | 40.9 ± 3.2 | 59.9 ± 0.6 |
| se | se_baseline | teacher | 3 | 46.5 ± 0.4 | 40.5 ± 1.1 | 37.4 ± 1.3 | 59.4 ± 0.2 |
| se | se_ours | student | 1 | 51.1 | 48.8 | 47.0 | 59.7 |
| se | se_ours | teacher | 1 | 51.6 | 49.7 | 48.1 | 59.4 |
| se | se_ours_emoji | student | 3 | 61.0 ± 0.7 | 61.0 ± 0.7 | 60.9 ± 0.8 | 61.4 ± 0.4 |
| se | se_ours_emoji | teacher | 2 | 61.1 ± 0.8 | 61.1 ± 0.8 | 61.3 ± 0.8 | 60.0 ± 0.6 |
| se | se_ours_gate | student | 3 | 62.2 ± 0.9 | 62.1 ± 0.9 | 62.3 ± 1.0 | 61.0 ± 0.7 |
| se | se_ours_hd | student | 3 | 45.6 ± 2.0 | 39.3 ± 4.1 | 36.2 ± 4.7 | 58.7 ± 0.2 |
| se | se_ours_hd | teacher | 3 | 49.9 ± 2.1 | 47.2 ± 3.2 | 45.4 ± 3.8 | 58.7 ± 0.6 |
| se | se_rkd | student | 3 | 59.6 ± 1.9 | 59.4 ± 2.1 | 59.0 ± 2.5 | 61.5 ± 0.5 |
| se | se_rkd | teacher | 3 | 60.2 ± 0.6 | 60.2 ± 0.5 | 60.0 ± 0.5 | 61.4 ± 0.6 |

### Slice: tweets WITHOUT emojis/hashtags

| dataset | run | model | runs | acc | macro_f1 | weighted_f1 | hate_f1 |
|---|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1 | 93.2 | 61.0 | 92.0 | 25.5 |
| dv | dv_baseline | teacher | 1 | 93.7 | 48.4 | 90.6 | 0.0 |
| dv | dv_ours_emoji | student | 1 | 88.7 | 67.9 | 90.5 | 42.0 |
| dv | dv_ours_emoji | teacher | 1 | 90.1 | 68.8 | 91.4 | 42.9 |
| dv | dv_ours_gate | student | 1 | 91.5 | 69.6 | 92.2 | 43.8 |
| dv | dv_ours_gate_cb | student | 1 | 89.3 | 68.6 | 90.9 | 43.1 |
| dv | dv_ours_hd | student | 1 | 91.0 | 70.3 | 92.0 | 45.4 |
| dv | dv_ours_hd | teacher | 1 | 91.6 | 66.7 | 91.9 | 38.0 |
| dv | dv_rkd | student | 1 | 92.4 | 70.6 | 92.7 | 45.4 |
| dv | dv_rkd | teacher | 1 | 91.2 | 68.2 | 91.8 | 41.1 |
| se | se_abl_emo_no_cb | student | 1 | 54.2 | 51.8 | 50.1 | 62.6 |
| se | se_abl_emo_no_gate | student | 1 | 50.3 | 45.5 | 43.1 | 61.7 |
| se | se_abl_emo_no_revise | student | 1 | 52.4 | 48.7 | 46.6 | 62.5 |
| se | se_abl_emoji_paperkd | student | 1 | 50.7 | 46.6 | 44.3 | 61.3 |
| se | se_abl_emoji_paperkd | teacher | 1 | 58.7 | 58.3 | 57.6 | 62.7 |
| se | se_abl_kd_only | student | 1 | 56.0 | 54.2 | 52.8 | 63.4 |
| se | se_abl_spn_paperkd | student | 1 | 58.5 | 57.5 | 56.6 | 63.9 |
| se | se_abl_spnhd_paperkd | student | 1 | 54.3 | 52.2 | 50.6 | 62.3 |
| se | se_baseline | student | 3 | 51.8 ± 4.7 | 46.9 ± 7.5 | 44.5 ± 8.4 | 62.4 ± 1.2 |
| se | se_baseline | teacher | 3 | 45.9 ± 0.9 | 37.4 ± 1.6 | 33.9 ± 1.8 | 60.5 ± 0.4 |
| se | se_ours | student | 1 | 60.7 | 60.3 | 59.7 | 64.1 |
| se | se_ours | teacher | 1 | 61.8 | 61.7 | 61.5 | 63.4 |
| se | se_ours_emoji | student | 3 | 53.2 ± 3.2 | 50.2 ± 4.6 | 48.4 ± 5.2 | 62.1 ± 0.9 |
| se | se_ours_emoji | teacher | 2 | 59.4 ± 1.4 | 59.2 ± 1.5 | 58.8 ± 1.7 | 62.2 ± 0.5 |
| se | se_ours_gate | student | 3 | 58.8 ± 3.8 | 57.8 ± 5.1 | 56.9 ± 5.8 | 63.3 ± 0.6 |
| se | se_ours_hd | student | 3 | 54.2 ± 4.8 | 51.0 ± 7.3 | 49.2 ± 8.2 | 62.7 ± 1.2 |
| se | se_ours_hd | teacher | 3 | 59.5 ± 4.2 | 58.6 ± 4.9 | 58.0 ± 5.6 | 62.7 ± 0.4 |
| se | se_rkd | student | 3 | 52.0 ± 5.1 | 47.8 ± 7.9 | 45.7 ± 8.8 | 61.9 ± 1.7 |
| se | se_rkd | teacher | 3 | 56.7 ± 1.3 | 56.0 ± 1.4 | 55.2 ± 1.5 | 61.6 ± 1.0 |

### Cross-dataset generalisation

| dataset | run | model | runs | acc | macro_f1 | weighted_f1 | hate_f1 | ece |
|---|---|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1 | 58.1 | 38.4 | 43.9 | 3.6 | 41.1 |
| dv | dv_baseline | teacher | 1 | 57.8 | 36.6 | 42.4 | 0.0 | 36.9 |
| dv | dv_ours_emoji | student | 1 | 57.4 | 45.2 | 49.2 | 19.3 | 7.1 |
| dv | dv_ours_emoji | teacher | 1 | 57.7 | 43.9 | 48.3 | 16.2 | 11.1 |
| dv | dv_ours_gate | student | 1 | 57.6 | 41.1 | 46.0 | 10.0 | 14.1 |
| dv | dv_ours_gate_cb | student | 1 | 56.9 | 43.0 | 47.4 | 14.9 | 11.0 |
| dv | dv_ours_hd | student | 1 | 58.0 | 42.1 | 46.9 | 11.9 | 7.2 |
| dv | dv_ours_hd | teacher | 1 | 57.8 | 46.5 | 50.4 | 21.9 | 11.0 |
| dv | dv_rkd | student | 1 | 57.5 | 38.9 | 44.2 | 5.3 | 10.2 |
| dv | dv_rkd | teacher | 1 | 56.9 | 39.8 | 44.8 | 7.7 | 14.8 |
| se | se_abl_emo_no_cb | student | 1 | 49.9 | 36.5 | 62.3 | 7.4 | 8.0 |
| se | se_abl_emo_no_gate | student | 1 | 44.1 | 33.8 | 56.9 | 7.8 | 12.6 |
| se | se_abl_emo_no_revise | student | 1 | 48.3 | 35.7 | 60.9 | 7.3 | 9.2 |
| se | se_abl_emoji_paperkd | student | 1 | 48.5 | 35.8 | 61.1 | 7.2 | 13.4 |
| se | se_abl_emoji_paperkd | teacher | 1 | 49.1 | 36.6 | 61.5 | 8.4 | 26.5 |
| se | se_abl_kd_only | student | 1 | 42.9 | 33.1 | 55.7 | 7.6 | 23.3 |
| se | se_abl_spn_paperkd | student | 1 | 58.3 | 39.7 | 69.4 | 6.1 | 8.9 |
| se | se_abl_spnhd_paperkd | student | 1 | 48.4 | 35.8 | 60.9 | 7.3 | 16.2 |
| se | se_baseline | student | 3 | 50.0 ± 6.5 | 35.9 ± 3.1 | 62.2 ± 5.7 | 6.2 ± 0.1 | 29.4 ± 5.3 |
| se | se_baseline | teacher | 3 | 43.2 ± 0.9 | 32.7 ± 0.6 | 56.2 ± 0.8 | 6.1 ± 0.5 | 39.1 ± 6.1 |
| se | se_ours | student | 1 | 54.2 | 38.4 | 66.0 | 7.1 | 13.6 |
| se | se_ours | teacher | 1 | 54.5 | 38.4 | 66.3 | 6.9 | 20.8 |
| se | se_ours_emoji | student | 3 | 48.9 ± 2.1 | 36.3 ± 1.2 | 61.4 ± 1.9 | 8.0 ± 0.5 | 11.3 ± 1.7 |
| se | se_ours_emoji | teacher | 2 | 52.8 ± 3.6 | 38.0 ± 1.6 | 64.7 ± 3.1 | 7.8 ± 0.1 | 16.2 ± 3.2 |
| se | se_ours_gate | student | 3 | 52.0 ± 1.5 | 37.6 ± 0.8 | 64.1 ± 1.2 | 7.7 ± 0.3 | 12.2 ± 2.6 |
| se | se_ours_hd | student | 3 | 48.7 ± 4.2 | 35.9 ± 2.2 | 61.2 ± 3.7 | 7.3 ± 0.5 | 12.2 ± 3.3 |
| se | se_ours_hd | teacher | 3 | 54.0 ± 6.1 | 38.1 ± 2.7 | 65.7 ± 4.9 | 6.8 ± 0.4 | 15.6 ± 5.8 |
| se | se_rkd | student | 3 | 43.2 ± 4.6 | 33.1 ± 2.3 | 55.9 ± 4.7 | 7.3 ± 0.4 | 17.4 ± 5.8 |
| se | se_rkd | teacher | 3 | 45.2 ± 1.6 | 34.2 ± 0.8 | 57.9 ± 1.6 | 7.4 ± 0.2 | 27.2 ± 2.1 |

### Efficiency

| dataset | run | model | params (M) | size (MB) | latency (ms) | device |
|---|---|---|---|---|---|---|
| dv | dv_baseline | student | 1.45 | 5.6 | 2.24 | cpu |
| dv | dv_baseline | teacher | 5.31 | 20.3 | 4.44 | cpu |
| dv | dv_ours_emoji | student | 1.45 | 5.6 | 1.91 | cpu |
| dv | dv_ours_emoji | teacher | 5.31 | 20.3 | 3.88 | cpu |
| dv | dv_ours_gate_cb | student | 1.45 | 5.6 | 1.77 | cpu |
| dv | dv_ours_gate | student | 1.45 | 5.6 | 1.65 | cpu |
| dv | dv_ours_hd | student | 1.45 | 5.6 | 1.80 | cpu |
| dv | dv_ours_hd | teacher | 5.31 | 20.3 | 4.11 | cpu |
| dv | dv_rkd | student | 1.45 | 5.6 | 1.67 | cpu |
| dv | dv_rkd | teacher | 5.31 | 20.3 | 4.22 | cpu |
| se | se_abl_emo_no_cb | student | 1.45 | 5.6 | 1.96 | cpu |
| se | se_abl_emo_no_gate | student | 1.45 | 5.6 | 1.84 | cpu |
| se | se_abl_emo_no_revise | student | 1.45 | 5.6 | 1.94 | cpu |
| se | se_abl_emoji_paperkd | student | 1.45 | 5.6 | 1.91 | cpu |
| se | se_abl_emoji_paperkd | teacher | 5.31 | 20.3 | 3.86 | cpu |
| se | se_abl_kd_only | student | 1.45 | 5.6 | 2.48 | cpu |
| se | se_abl_spn_paperkd | student | 1.45 | 5.6 | 1.94 | cpu |
| se | se_abl_spnhd_paperkd | student | 1.45 | 5.6 | 2.01 | cpu |
| se | se_baseline | student | 1.45 | 5.6 | 1.92 | cpu |
| se | se_baseline | teacher | 5.31 | 20.3 | 5.17 | cpu |
| se | se_ours_emoji | student | 1.45 | 5.6 | 1.80 | cpu |
| se | se_ours_emoji | teacher | 5.31 | 20.3 | 4.01 | cpu |
| se | se_ours_gate | student | 1.45 | 5.6 | 2.08 | cpu |
| se | se_ours_hd | student | 1.45 | 5.6 | 1.84 | cpu |
| se | se_ours_hd | teacher | 5.31 | 20.3 | 4.68 | cpu |
| se | se_ours | student | 1.45 | 5.6 | 2.23 | cpu |
| se | se_ours | teacher | 5.31 | 20.3 | 4.72 | cpu |
| se | se_rkd | student | 1.45 | 5.6 | 1.86 | cpu |
| se | se_rkd | teacher | 5.31 | 20.3 | 3.89 | cpu |

### McNemar: se_rkd vs se_ours_gate (student)

| seed | only se_rkd correct | only se_ours_gate correct | chi2 | p | verdict (0.05) |
|---|---|---|---|---|---|
| 1 | 231 | 290 | 6.46 | 0.0111 | se_ours_gate significantly better |
| 2 | 298 | 557 | 77.85 | 0.0000 | se_ours_gate significantly better |
| 42 | 176 | 237 | 8.72 | 0.0032 | se_ours_gate significantly better |

### McNemar: se_rkd vs se_ours_emoji (student)

| seed | only se_rkd correct | only se_ours_emoji correct | chi2 | p | verdict (0.05) |
|---|---|---|---|---|---|
| 1 | 320 | 226 | 15.84 | 0.0001 | se_rkd significantly better |
| 2 | 221 | 421 | 61.68 | 0.0000 | se_ours_emoji significantly better |
| 42 | 171 | 184 | 0.41 | 0.5242 | no significant difference |

### McNemar: se_rkd vs se_ours_hd (student)

| seed | only se_rkd correct | only se_ours_hd correct | chi2 | p | verdict (0.05) |
|---|---|---|---|---|---|
| 1 | 683 | 245 | 205.79 | 0.0000 | se_rkd significantly better |
| 2 | 445 | 441 | 0.01 | 0.9197 | no significant difference |
| 42 | 509 | 264 | 77.02 | 0.0000 | se_rkd significantly better |

### McNemar: se_baseline vs se_ours_gate (student)

| seed | only se_baseline correct | only se_ours_gate correct | chi2 | p | verdict (0.05) |
|---|---|---|---|---|---|
| 1 | 277 | 729 | 202.19 | 0.0000 | se_ours_gate significantly better |
| 2 | 293 | 539 | 72.15 | 0.0000 | se_ours_gate significantly better |
| 42 | 318 | 621 | 97.13 | 0.0000 | se_ours_gate significantly better |