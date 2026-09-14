# l4/seeds/last4_data17 (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.3227 | 0.9381 | 88.5% |
| TabFM fine-tuned (last 4 ICL blocks, step 25) | 0.3246 | 0.9383 | 88.5% |
| Generator's true probabilities (ceiling) | 0.1916 | 0.9523 | 95.2% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | +0.0019 | [+0.0012, +0.0025] |
| accuracy | -0.0333 pts | [-0.2333, +0.1667] |
| roc_auc | +0.0001 | [-0.0000, +0.0003] |
