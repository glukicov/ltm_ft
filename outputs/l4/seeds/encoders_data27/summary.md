# l4/seeds/encoders_data27 (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6925 | 0.5214 | 51.8% |
| TabFM fine-tuned (last 4 ICL blocks + encoders, step 25) | 0.6933 | 0.4938 | 49.5% |
| Generator's true probabilities (ceiling) | 0.1906 | 0.9527 | 95.3% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | +0.0008 | [-0.0002, +0.0017] |
| accuracy | -2.3000 pts | [-4.4000, -0.1992] |
| roc_auc | -0.0276 | [-0.0490, -0.0042] |
