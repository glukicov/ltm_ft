# l4/checkerboard (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6232 | 0.7379 | 66.5% |
| TabFM fine-tuned (last 4 ICL blocks, step 100) | 0.6065 | 0.7468 | 66.9% |
| Generator's true probabilities (ceiling) | 0.2016 | 0.9490 | 94.9% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | -0.0167 | [-0.0198, -0.0136] |
| accuracy | +0.4667 pts | [-0.8675, +1.8333] |
| roc_auc | +0.0088 | [+0.0006, +0.0170] |
