# infomax-sbdr


TODO: test using cosine distance and normalization with 1e0 bias on the norm

## To comment in meetings

- AND similarity + L1 norm have much worse utilization of units, but probably better classification (because we directly train them to be linerarly separable). Test with noise?
- SHould we test ith autoencoder? On ImageNet would be not sane to use an autoencoder, should the test be standardized and same for all dataset?

## Training checklist

- l1 norm
- reconstruction loss
- FLO function (with or without exponential)
- weight decay value of adamw (0.00001)
- similarity function

## Validation checklist

- binarization

## Plots

Use error bars in all plots

Concentration:
    x: different concentration
    multiple traces: different average number of non-zero units
    use sharp activations 

Sharpness: AND vs Jaccard (with some fixed amount) 
    x: different sharpness
    multiple traces: different average number of non-zero units
    use high concentration

Ideal case:
    x: average number of non-zero units
    only two traces, ideal and not
    use same color  but different patterns for ideal vs non-ideal

Resistance to noise
    x: amount of noise
    multiple traces: different average of non-zero units
    use high concentration
    use sharp activation

