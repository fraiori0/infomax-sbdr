- bias 1e-2
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

Global Pooling: AVG <-- remember to change in conv_modules.py

## Unit statistics

Statistics on unit activity
        Per-unit quantiles: [0.00467163 0.00545591 0.00671853 0.00856788 0.01116039 0.01466969
 0.01917658 0.02363391 0.03274735]
        Per-sample quantiles: [0.00491407 0.0067874  0.00790812 0.00971777 0.01213809 0.01484997
 0.01770266 0.01970642 0.02400858]

Histograms
        Bin Count:
                [ 0.  0.  1.  0.  7.  9.  8. 16. 25. 27. 40. 26. 36. 20. 16.  6. 10.  1. 6.  1.]
        Bin Centers:
                [0.00278118 0.00318551 0.00364862 0.00417906 0.00478661 0.00548249
 0.00627954 0.00719246 0.00823811 0.00943577 0.01080754 0.01237875
 0.01417838 0.01623964 0.01860056 0.02130472 0.02440201 0.02794958
 0.0320129  0.7670911 ]

Sharpness of unit activity:
        Low: [0.96332115 0.97541463 0.97962636 0.9819248 ]
        Middle: [0.03412286 0.02002362 0.01455394 0.01134542]
        High: [0.00255599 0.00456176 0.00581966 0.00672982]

Unused units:
        Threshold 0.001
          [0 0 0 0 0]
        Threshold 0.010
          [0 0 0 0 0]
        Threshold 0.050
          [0 0 0 0 0]
        Threshold 0.100
          [0 0 1 1 1]
        Threshold 0.150
          [1 1 1 1 1]

## CIFAR10

## CIFAR100