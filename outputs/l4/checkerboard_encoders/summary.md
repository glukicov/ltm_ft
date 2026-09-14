# l4/checkerboard_encoders (checkerboard)

3,000 held-out rows. Lower log loss is better; higher ROC AUC and accuracy are better.

| Model | Log loss | ROC AUC | Accuracy |
|---|---|---|---|
| TabFM zero-shot | 0.6232 | 0.7379 | 66.5% |
| TabFM fine-tuned (last 4 ICL blocks + encoders, step 75) | 0.2237 | 0.9485 | 93.8% |
| Generator's true probabilities (ceiling) | 0.2016 | 0.9490 | 94.9% |

Fine-tuned minus zero-shot, with 95% paired-bootstrap intervals over test rows:

| Metric | Δ | 95% interval |
|---|---|---|
| log_loss | -0.3994 | [-0.4223, -0.3744] |
| accuracy | +27.3333 pts | [+25.5333, +29.1008] |
| roc_auc | +0.2105 | [+0.1927, +0.2273] |
