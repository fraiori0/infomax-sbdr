## Parameters

- log(sim()) critic i.e. "unmodified" flo
- L1 norm regularization 


- good classification performances with linear SVM

## CIFAR10

Statistics on unit activity
        Per-unit quantiles: [3.9713144e-09 4.4664201e-09 6.0362142e-09 1.0510438e-08 4.9268046e-08
 1.3041014e-03 3.5653257e-01 4.3168557e-01 5.7648659e-01]
        Per-sample quantiles: [0.06265478 0.06493822 0.06616562 0.06827658 0.07067405 0.07302344
 0.07521209 0.07655974 0.07923225]

Histograms
        Bin Count:
                [111.  21.   9.   9.   4.   3.   3.   6.   4.   5.   7.   8.   3.   1.
   5.   0.   6.  16.  33.   1.]
        Bin Centers:
                [1.4280047e-08 5.1862852e-08 1.4761964e-07 3.9159724e-07 1.0132241e-06
 2.5970598e-06 6.6324974e-06 1.6914340e-05 4.3111329e-05 1.0985833e-04
 2.7992213e-04 7.1322540e-04 1.8172346e-03 4.6301251e-03 1.1797059e-02
 3.0057596e-02 7.6583378e-02 1.9512598e-01 4.9715906e-01 1.1070309e+00]

Sharpness of unit activity:
        Low: [0.8372819 0.8755648 0.8922389 0.9020796]
        Middle: [0.14170758 0.09236868 0.06898    0.05437807]
        High: [0.02101051 0.0320665  0.03878106 0.04354232]

Unused units:
        Threshold 0.001
          [153 154 157 158 159]
        Threshold 0.010
          [163 163 167 167 170]
        Threshold 0.050
          [170 171 174 180 182]
        Threshold 0.100
          [178 181 185 188 191]
        Threshold 0.150
          [185 186 189 193 195]

### Sparse K

#### K = 7

K-Nearest Neighbors Classification
        Distances shape: (8000, 42000)
        Nearest indices shape: (8000, 19)
        Nearest labels shape: (8000, 19, 10)
        Correct mask shape: (8000, 19)
        KNN accuracy: 0.4472236633300781
        KNN accuracy per example: 0.4472237229347229,  std: 0.3264833688735962
        Average label shape: (8000, 10)
[[0.         1.         0.         0.         0.         0.
  0.         0.         0.         0.        ]
 [0.         0.         0.         0.         0.         0.
  1.         0.         0.         0.        ]
 [0.15789473 0.         0.         0.         0.         0.
  0.         0.         0.84210527 0.        ]
 [0.         0.         0.10526316 0.05263158 0.36842105 0.05263158
  0.05263158 0.36842105 0.         0.        ]
 [0.15789473 0.         0.36842105 0.21052632 0.         0.15789473
  0.         0.05263158 0.05263158 0.        ]]
        KNN accuracy (vote)
                K=19 : 0.5308

Linear Support Vector Classification
        Labels train shape (categorical): (42000,)
        Labels val shape (categorical): (8000,)

Training Linear SVM on the training set
          Time: 1.81 seconds
        Accuracy on training set: 0.5332380952380953
        Accuracy on validation set: 0.530875


## CIFAR100

Linear Support Vector Classification
    Coarse labels
        Time: 291.81 seconds
    Accuracy
        Train: 0.31547619047619047
        Valid: 0.3095
    Fine labels
        Time: 713.20 seconds
    Accuracy
        Train: 0.22154761904761905
        Valid: 0.213875

### Sparse K

#### K = 7

Linear Support Vector Classification
        Coarse labels
          Time: 5.70 seconds
        Accuracy
          Train: 0.2465952380952381
          Valid: 0.242625
        Fine labels
          Time: 19.67 seconds
        Accuracy
          Train: 0.15007142857142858
          Valid: 0.1455