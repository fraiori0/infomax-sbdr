# AntiHebbian Learning on Temporal Differences

## Ablation on TD

When testing a module of 256 units (FashionMNIST) by applying the forward weights to either:

- the linear TD prediction of the input (TD), or 
- the past discounted sum of the input (no-TD)

the TD model performs significantly better in term of classification accuracy and its robustness to sparsification.