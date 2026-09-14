# l4/seeds/last4_data27 (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6925 | 0.5214 | 51.8% |
| TabFM fine-tuned (last 4 ICL blocks, step 125) | 0.6932 | 0.5007 | 49.5% |
| Generator's true probabilities (ceiling) | 0.1906 | 0.9527 | 95.3% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | +0.0007 | [-0.0004, +0.0018] |
| accuracy | -2.2667 pts | [-5.2008, +0.5667] |
| roc_auc | -0.0206 | [-0.0557, +0.0123] |
