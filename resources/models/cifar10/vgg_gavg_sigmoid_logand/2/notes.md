- bias 1e0
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

Global Pooling: AVG <-- remember to change in conv_modules.py

## Unit statistics

Statistics on unit activity
        Per-unit quantiles: [0.04111421 0.04655011 0.05069575 0.05705232 0.06548283 0.07396869
 0.08603507 0.09439798 0.10730157]
        Per-sample quantiles: [0.04686734 0.05296534 0.05552715 0.06009063 0.06598206 0.07250448
 0.07968848 0.08459317 0.09470376]

Histograms
        Bin Count:
                [ 3.  4.  4.  7. 12. 19. 20. 30. 24. 27. 33. 18. 15. 12.  8.  6.  5.  5.
  3.  0.]
        Bin Centers:
                [0.04026077 0.04257719 0.0450269  0.04761756 0.05035727 0.0532546
 0.05631864 0.05955897 0.06298573 0.06660967 0.07044208 0.07449502
 0.07878114 0.08331385 0.08810738 0.09317669 0.09853767 0.10420709
 0.11020271 0.8066422 ]

Sharpness of unit activity:
        Low: [0.9247362  0.9278983  0.92927134 0.9301145 ]
        Middle: [0.01600781 0.01021884 0.00762639 0.00601997]
        High: [0.05925599 0.0618829  0.06310226 0.06386554]

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