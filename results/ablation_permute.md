| Model | auroc_normal | auroc_permuted | delta | rank_normal | rank_permuted | acc_normal | acc_permuted |
|---|---|---|---|---|---|---|---|
| Logistic regression (pixels) | 0.8989 | 0.8987 | -0.0002 | 4 | 3 | 0.8042 | 0.8026 |
| Small CNN (scratch) | 0.9458 | 0.8562 | -0.0896 | 2 | 4 | 0.8285 | 0.7492 |
| ResNet-50 (ImageNet, fine-tuned) | 0.9717 | 0.9002 | -0.0715 | 1 | 2 | 0.8382 | 0.7621 |
| DenseNet-121 (ImageNet, fine-tuned) | 0.9233 | 0.9018 | -0.0215 | 3 | 1 | 0.8026 | 0.7476 |