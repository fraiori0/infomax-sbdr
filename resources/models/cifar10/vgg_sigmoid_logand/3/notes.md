## Parameters
- low bias (1e-2) and temperature
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

## CIFAR10

### Sparse K

#### K = 7

K-Nearest Neighbors Classification
        Distances shape: (8000, 42000)
        Nearest indices shape: (8000, 19)
        Nearest labels shape: (8000, 19, 10)
        Correct mask shape: (8000, 19)
        KNN accuracy: 0.32580921053886414
        KNN accuracy per example: 0.32580921053886414,  std: 0.2645208537578583
        Average label shape: (8000, 10)
[[0.         0.         0.         0.21052632 0.47368422 0.2631579
  0.         0.05263158 0.         0.        ]
 [0.         0.         0.         0.21052632 0.10526316 0.05263158
  0.36842105 0.15789473 0.05263158 0.05263158]
 [0.         0.         0.2631579  0.05263158 0.2631579  0.21052632
  0.10526316 0.         0.05263158 0.05263158]
 [0.         0.         0.10526316 0.05263158 0.05263158 0.
  0.7894737  0.         0.         0.        ]
 [0.         0.         0.         0.42105263 0.05263158 0.36842105
  0.         0.10526316 0.05263158 0.        ]]
        KNN accuracy (vote)
                K=19 : 0.4479

Linear Support Vector Classification
        Labels train shape (categorical): (42000,)
        Labels val shape (categorical): (8000,)

Training Linear SVM on the training set
          Time: 2.11 seconds
        Accuracy on training set: 0.5171904761904762
        Accuracy on validation set: 0.511625

## CIFAR100

### Sparse K

#### K = 7
