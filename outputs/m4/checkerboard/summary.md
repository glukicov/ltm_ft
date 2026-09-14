# m4/checkerboard (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6140 | 0.7642 | 69.1% |
| TabFM fine-tuned (last 4 ICL blocks, step 75) | 0.5934 | 0.7721 | 69.7% |
| Generator's true probabilities (ceiling) | 0.2016 | 0.9490 | 94.9% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | -0.0206 | [-0.0236, -0.0175] |
| accuracy | +0.5667 pts | [-0.5667, +1.8000] |
| roc_auc | +0.0079 | [+0.0024, +0.0137] |
