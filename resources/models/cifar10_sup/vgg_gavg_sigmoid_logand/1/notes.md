- bias 1e-2
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

Global Pooling: AVG <-- remember to change in conv_modules.py

## Unit statistics

Statistics on unit activity
        Per-unit quantiles: [0.00449287 0.00673244 0.00759167 0.01092128 0.01448283 0.0191201
 0.02459935 0.02641895 0.03507243]
        Per-sample quantiles: [0.00617743 0.00844035 0.00967705 0.01204215 0.01493315 0.01825245
 0.02177639 0.02403683 0.02946535]

Histograms
        Bin Count:
                [ 1.  0.  2.  0.  8.  5. 15. 15. 14. 24. 32. 30. 32. 31. 17. 17.  3.  5.
  3.  1.]
        Bin Centers:
                [0.00352951 0.00402247 0.00458428 0.00522455 0.00595425 0.00678587
 0.00773363 0.00881376 0.01004475 0.01144767 0.01304654 0.01486871
 0.01694538 0.01931209 0.02200935 0.02508334 0.02858664 0.03257925
 0.0371295  0.7697766 ]

Sharpness of unit activity:
        Low: [0.9485014 0.9669795 0.9733557 0.9767686]
        Middle: [0.04926163 0.02850242 0.02058268 0.01601693]
        High: [0.00223698 0.00451804 0.00606157 0.00721447]

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

## CIFAR100