# l4/seeds/last4_data37 (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6010 | 0.8437 | 75.5% |
| TabFM fine-tuned (last 4 ICL blocks, step 50) | 0.5852 | 0.8721 | 79.0% |
| Generator's true probabilities (ceiling) | 0.1996 | 0.9497 | 95.0% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | -0.0158 | [-0.0172, -0.0144] |
| accuracy | +3.5333 pts | [+2.5667, +4.5667] |
| roc_auc | +0.0284 | [+0.0242, +0.0331] |
