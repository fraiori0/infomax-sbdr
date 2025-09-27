- bias 1e-2
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

Global Pooling: MAX <-- remember to change in conv_modules.py


## CIFAR10

~49%

## CIFAR100