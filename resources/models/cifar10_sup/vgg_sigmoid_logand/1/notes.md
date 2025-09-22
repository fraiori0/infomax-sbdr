This time we measure contrastive loss only between samples of different classes

- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

## CIFAR10

Statistics on unit activity
        Per-unit quantiles: [0.00287306 0.00659225 0.00824789 0.01146131 0.01553454 0.01995724
 0.02838015 0.03571242 0.05134968]
        Per-sample quantiles: [0.00652897 0.00922287 0.01069945 0.0135044  0.01692371 0.02070393
 0.02455331 0.02698975 0.03228225]

Histograms
        Bin Count:
                [ 1.  2.  0.  1.  0.  5.  7. 13. 22. 29. 49. 54. 31. 18. 10. 11.  0.  1.
  1.  1.]
        Bin Centers:
                [0.00187074 0.00229168 0.00280733 0.00343902 0.00421284 0.00516078
 0.00632202 0.00774455 0.00948716 0.01162189 0.01423695 0.01744042
 0.02136473 0.02617205 0.03206107 0.03927518 0.04811254 0.05893844
 0.07220028 0.7897509 ]

Sharpness of unit activity:
        Low: [0.9489311  0.9658709  0.97180146 0.97501665]
        Middle: [0.04726415 0.02766276 0.02005078 0.01561311]
        High: [0.00380477 0.00646632 0.00814774 0.00937023]

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
        Distances shape: (5000, 45000)
        Nearest indices shape: (5000, 19)
        Nearest labels shape: (5000, 19, 10)
        Correct mask shape: (5000, 19)
        KNN accuracy: 0.43281054496765137
        KNN accuracy per example: 0.43281054496765137,  std: 0.3172522783279419
        Average label shape: (5000, 10)
[[0.         0.         0.2631579  0.         0.6315789  0.05263158
  0.05263158 0.         0.         0.        ]
 [0.         0.7368421  0.         0.         0.         0.
  0.         0.         0.05263158 0.21052632]
 [0.         0.         0.         0.         0.         0.
  0.         1.         0.         0.        ]
 [0.         0.         0.05263158 0.         0.36842105 0.47368422
  0.         0.10526316 0.         0.        ]
 [0.36842105 0.         0.05263158 0.05263158 0.05263158 0.
  0.         0.10526316 0.36842105 0.        ]]
        KNN accuracy (vote)
                K=19 : 0.5344

Linear Support Vector Classification
        Labels train shape (categorical): (45000,)
        Labels val shape (categorical): (5000,)

Training Linear SVM on the training set

(~53%, non cpiato per errore)