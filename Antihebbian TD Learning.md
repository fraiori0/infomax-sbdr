# AntiHebbian Learning on Temporal Differences


## Single Layer Encoder

### Linear Classification Accuracy

Using a linear SVM produce almost state-of-the-art results on the MNIST (95.2%, 1024 units) and FashionMNIST (84.5%, 1024 units).
Also, accuracy kept increasing with the number of units, scaling (128>256>1024). 
We should still check what happend when a single layer is scaled further.

### Ablation on TD

When testing a module of 256 units (FashionMNIST) by applying the forward weights to either:

- the linear TD prediction of the input (TD), or 
- the past discounted sum of the input (no-TD)

the TD model performs significantly better in term of classification accuracy and its robustness to sparsification.