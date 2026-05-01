# Full Evaluation Report

Generated: 2026-04-22 02:16:35

## System-Level Detection (vs ground_truth_events.csv)

Run: `D:\school\uni\sem 6\cv\traffic\runs\20260422_020904`

| Metric          | Value |
|-----------------|-------|
| GT events       | 127 |
| Pred events     | 7 |
| True positives  | 0 |
| False negatives | 127 |
| False positives | 7 |
| **Precision**   | **0.0%** |
| **Recall**      | **0.0%** |
| **F1**          | **0.0%** |
| Lane agreement  | 0.0% |
| Class agreement | 0.0% |

## Appearance Model Comparison

| Model | Samples | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ganomaly | 600 | 97.0% | 75.3% | 84.8% | 0.905 | 0.936 |
| vae | 600 | 93.9% | 51.0% | 66.1% | 0.805 | 0.849 |

### ganomaly — per class group

| Class | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| bus | 93.8% | 75.0% | 83.3% | 0.982 | 0.983 |
| car | 98.6% | 60.9% | 75.3% | 0.801 | 0.848 |
| truck | 96.9% | 86.9% | 91.6% | 0.972 | 0.980 |

### vae — per class group

| Class | Precision | Recall | F1 | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| bus | 92.3% | 60.0% | 72.7% | 0.941 | 0.941 |
| car | 94.9% | 48.7% | 64.4% | 0.734 | 0.800 |
| truck | 93.6% | 50.3% | 65.5% | 0.818 | 0.857 |
