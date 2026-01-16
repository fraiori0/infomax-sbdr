- high bias 1e0
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)


## CIFAR10

Statistics on unit activity
        Per-unit quantiles: [0.02849532 0.03239576 0.03712869 0.05321767 0.06975623 0.08593621
 0.10491819 0.12696071 0.14687657]
        Per-sample quantiles: [0.04763812 0.05518484 0.05861177 0.0639796  0.0703235  0.07782731
 0.08535808 0.09028092 0.10040223]

Histograms
        Bin Count:
                [ 1.  8.  5. 11.  7.  6. 16. 24. 22. 24. 32. 30. 24. 16. 10.  4.  6.  7. 1.  1.]
        Bin Centers:
                [0.0261283  0.02890429 0.03197521 0.03537239 0.0391305  0.0432879
                0.04788699 0.05297471 0.05860296 0.06482919 0.07171692 0.07933643
                0.08776548 0.09709005 0.10740531 0.11881651 0.13144007 0.14540485
                0.16085327 0.83448356]

Sharpness of unit activity:
        Low: [0.9159283 0.9207821 0.922881  0.9241874]
        Middle: [0.02411096 0.01544299 0.01154027 0.00910305]
        High: [0.05996075 0.06377493 0.06557868 0.06670954]

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


K-Nearest Neighbors Classification
        Distances shape: (8000, 42000)
        Nearest indices shape: (8000, 50)
        Nearest labels shape: (8000, 50, 10)
        Correct mask shape: (8000, 50)
        KNN accuracy: 0.3712475001811981
        KNN accuracy per example: 0.37124747037887573,  std: 0.24198314547538757
        Average label shape: (8000, 10)
        KNN accuracy (vote)
                K=50 : 0.5134

Linear Support Vector Classification
        Labels train shape (categorical): (42000,)
        Labels val shape (categorical): (8000,)

Training Linear SVM on the training set
          Time: 5.20 seconds
        Accuracy on training set: 0.518452380952381
        Accuracy on validation set: 0.514625

## CIFAR100