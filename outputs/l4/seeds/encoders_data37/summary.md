# l4/seeds/encoders_data37 (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6010 | 0.8437 | 75.5% |
| TabFM fine-tuned (last 4 ICL blocks + encoders, step 150) | 0.2193 | 0.9495 | 94.1% |
| Generator's true probabilities (ceiling) | 0.1996 | 0.9497 | 95.0% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | -0.3817 | [-0.4048, -0.3587] |
| accuracy | +18.6333 pts | [+17.1333, +20.1333] |
| roc_auc | +0.1058 | [+0.0935, +0.1187] |
