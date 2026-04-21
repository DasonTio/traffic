# Full Evaluation Report

Generated: 2026-04-22 02:32:34

## System-Level Detection (vs ground_truth_events.csv)

Run: `D:\school\uni\sem 6\cv\traffic\runs\20260422_022848`

| Metric          | Value |
|-----------------|-------|
| GT events       | 127 |
| Pred events     | 6 |
| True positives  | 0 |
| False negatives | 127 |
| False positives | 6 |
| **Precision**   | **0.0%** |
| **Recall**      | **0.0%** |
| **F1**          | **0.0%** |
| Lane agreement  | 0.0% |
| Class agreement | 0.0% |

## Appearance Model Comparison

| Model | Samples | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ganomaly | 600 | 97.0% | 75.3% | 84.8% | 0.905 | 0.936 |
| vae | 600 | 92.9% | 52.0% | 66.7% | 0.803 | 0.846 |

### ganomaly — per class group

| Class | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| bus | 93.8% | 75.0% | 83.3% | 0.982 | 0.983 |
| car | 98.6% | 60.9% | 75.3% | 0.801 | 0.848 |
| truck | 96.9% | 86.9% | 91.6% | 0.972 | 0.980 |

### vae — per class group

| Class | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| bus | 88.9% | 60.0% | 71.6% | 0.943 | 0.943 |
| car | 94.8% | 47.8% | 63.6% | 0.735 | 0.800 |
| truck | 92.8% | 53.1% | 67.5% | 0.815 | 0.851 |
