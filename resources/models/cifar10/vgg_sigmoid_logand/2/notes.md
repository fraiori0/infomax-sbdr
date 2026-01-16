## Parameters

- and with a bias of 1, used with the flo estimator without exponentiation (i.e., like a log cancels out)

## Statistics

Statistics on unit activity
        Per-unit quantiles: [0.04376099 0.04746464 0.04989217 0.05625718 0.06481232 0.07529236
 0.09039955 0.10652751 0.13172781]
        Per-sample quantiles: [0.04743082 0.05430208 0.05695664 0.06168544 0.0668222  0.07421727
 0.08166399 0.08649717 0.09700358]

Histograms
        Bin Count:
                [ 0.  0.  6. 14. 22. 25. 29. 39. 36. 24. 25.  4.  7.  7.  5.  4.  1.  4.
  2.  1.]
        Bin Centers:
                [0.03819753 0.04105358 0.04412318 0.0474223  0.05096809 0.05477898
 0.05887484 0.06327695 0.0680082  0.07309321 0.07855843 0.08443229
 0.09074533 0.09753039 0.10482279 0.11266045 0.12108412 0.13013765
 0.13986811 0.82245433]

Sharpness of unit activity:
        Low: [0.9227058  0.9261039  0.92758274 0.92849624]
        Middle: [0.01713914 0.01092959 0.00816527 0.00642327]
        High: [0.06015504 0.06296652 0.06425195 0.06508045]

Unused units:
        Threshold 0.001
          [0 0 0 0 0]
        Threshold 0.010
          [0 0 0 0 0]
        Threshold 0.050
          [0 0 0 0 0]
        Threshold 0.100
          [0 0 0 0 0]
        Threshold 0.150
          [0 0 0 0 0]

## CIFAR10

K-Nearest Neighbors Classification
        Distances shape: (8000, 42000)
        Nearest indices shape: (8000, 50)
        Nearest labels shape: (8000, 50, 10)
        Correct mask shape: (8000, 50)
        KNN accuracy: 0.43353497982025146
        KNN accuracy per example: 0.43353497982025146,  std: 0.2923913300037384
        Average label shape: (8000, 10)
        KNN accuracy (vote)
                K=50 : 0.5445

Linear Support Vector Classification
        Labels train shape (categorical): (42000,)
        Labels val shape (categorical): (8000,)

Training Linear SVM on the training set
          Time: 6.61 seconds
        Accuracy on training set: 0.5666428571428571
        Accuracy on validation set: 0.560625

### Sparse K

#### K = 7

Binarizing/Sparsifying the encodings
        Binarization threshold: None
        Maximum number of non-zero elements: 7

Training Linear SVM on the training set
          Time: 2.86 seconds
        Accuracy on training set: 0.5413333333333333
        Accuracy on validation set: 0.540625

## CIFAR100

Linear Support Vector Classification
        Coarse labels
          Time: 108.49 seconds
        Accuracy
          Train: 0.27635714285714286
          Valid: 0.253125
        Fine labels
          Time: 495.85 seconds
        Accuracy
          Train: 0.1894047619047619
          Valid: 0.13975

### Sparse K

#### K = 7

Linear Support Vector Classification
        Coarse labels
          Time: 7.23 seconds
        Accuracy
          Train: 0.24769047619047618
          Valid: 0.218125
        Fine labels
          Time: 21.41 seconds
        Accuracy
          Train: 0.15128571428571427
          Valid: 0.11775