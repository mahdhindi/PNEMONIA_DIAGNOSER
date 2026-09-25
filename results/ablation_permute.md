| Model | auroc_normal | auroc_permuted | delta | rank_normal | rank_permuted | acc_normal | acc_permuted |
|---|---|---|---|---|---|---|---|
| Logistic regression (pixels) | 0.8950 | 0.8949 | -0.0001 | 4 | 2 | 0.7743 | 0.7727 |
| Small CNN (scratch) | 0.9416 | 0.8463 | -0.0952 | 2 | 4 | 0.8261 | 0.7605 |
| ResNet-50 (ImageNet, fine-tuned) | 0.9706 | 0.8926 | -0.0780 | 1 | 3 | 0.8625 | 0.7743 |
| DenseNet-121 (ImageNet, fine-tuned) | 0.9357 | 0.9088 | -0.0269 | 3 | 1 | 0.8220 | 0.7605 |