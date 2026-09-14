# l4/runners (runners)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.5266 | 0.8240 | 74.8% |
| TabFM fine-tuned (last 4 ICL blocks, step 0) | 0.5266 | 0.8240 | 74.8% |
| Generator's true probabilities (ceiling) | 0.4732 | 0.8537 | 77.8% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | +0.0000 | [+0.0000, +0.0000] |
| accuracy | +0.0000 pts | [+0.0000, +0.0000] |
| roc_auc | +0.0000 | [+0.0000, +0.0000] |
