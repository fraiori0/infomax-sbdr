This time we measure contrastive loss only between samples of different classes

- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

!!!
Trained using average similarity with same class, and contrast inter-class

## CIFAR10


Semantic Labels
        Per-label count: [4531. 4509. 4502. 4474. 4494. 4507. 4533. 4503. 4476. 4471.]
        Average per-label activity: [0.0162065  0.02003503 0.01566619 0.01582319 0.01561072 0.01564295
 0.0176827  0.01827018 0.01915785 0.02060873]
        Semantic labels shape: (10, 256)

Average activity: 0.017560791224241257

Forward pass on the whole validation set
100%|███████████████████████████████████████████████████████| 20/20 [00:02<00:00,  6.93it/s]
        Encoding shape (zs_val): (5000, 256)
        Labels shape (one-hot): (5000, 10)
        Labels shape (categorical): (5000,)

Statistics on unit activity
        Per-unit quantiles: [0.00262084 0.00559301 0.00785477 0.01080596 0.01448389 0.02118319
 0.02802999 0.03931553 0.05786853]
        Per-sample quantiles: [0.0060995  0.00877144 0.01035855 0.01321503 0.01689638 0.02103356
 0.02533204 0.02798286 0.03343971]

Histograms
        Bin Count:
                [ 0.  1.  0.  0.  0.  0.  0.  1.  0.  1.  7. 11. 39. 79. 55. 41. 10.  9.
  1.  0.]
        Bin Centers:
                [1.0553970e-04 1.5270125e-04 2.2093540e-04 3.1965776e-04 4.6249101e-04
 6.6914479e-04 9.6813531e-04 1.4007203e-03 2.0265914e-03 2.9321136e-03
 4.2422395e-03 6.1377524e-03 8.8802148e-03 1.2848059e-02 1.8588813e-02
 2.6894633e-02 3.8911633e-02 5.6298070e-02 8.1453077e-02 7.9816371e-01]

Sharpness of unit activity:
        Low: [0.96007633 0.97078925 0.9747056  0.9768647 ]
        Middle: [0.03355816 0.02019965 0.01476224 0.01154991]
        High: [0.00636554 0.00901111 0.0105322  0.01158542]

Unused units:
        Threshold 0.001
          [0 0 0 0 0]
        Threshold 0.010
          [0 0 0 0 0]
        Threshold 0.050
          [0 0 0 1 1]
        Threshold 0.100
          [0 0 1 1 2]
        Threshold 0.150
          [1 1 1 2 2]

K-Nearest Neighbors Classification
        Distances shape: (5000, 45000)
        Nearest indices shape: (5000, 19)
        Nearest labels shape: (5000, 19, 10)
        Correct mask shape: (5000, 19)
        KNN accuracy: 0.5497684478759766
        KNN accuracy per example: 0.5497683882713318,  std: 0.3695795238018036
        Average label shape: (5000, 10)
[[0.         0.         0.31578946 0.         0.6315789  0.05263158
  0.         0.         0.         0.        ]
 [0.10526316 0.5263158  0.         0.         0.05263158 0.
  0.         0.05263158 0.15789473 0.10526316]
 [0.         0.         0.         0.         0.         0.
  0.         1.         0.         0.        ]
 [0.         0.         0.7368421  0.10526316 0.         0.05263158
  0.10526316 0.         0.         0.        ]
 [0.36842105 0.05263158 0.05263158 0.21052632 0.21052632 0.
  0.         0.         0.10526316 0.        ]]
        KNN accuracy (vote)
                K=19 : 0.6278

Linear Support Vector Classification
        Labels train shape (categorical): (45000,)
        Labels val shape (categorical): (5000,)

Training Linear SVM on the training set
          Time: 65.44 seconds
        Accuracy on training set: 0.6588222222222222
        Accuracy on validation set: 0.6428


## CIFAR100

