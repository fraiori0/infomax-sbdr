- bias 1e-2
- double layer negpmi head
- log-canceled FLO (we use logand with the "modified" FLO, without the exp, so we don't even do the log)

Global Pooling: AVG <-- remember to change in conv_modules.py

## Unit statistics

Statistics on unit activity
        Per-unit quantiles: [2.6038668e-07 4.8481377e-07 7.8041029e-07 7.2573343e-06 1.0997708e-02
 4.2218897e-01 6.0407054e-01 7.1743685e-01 7.9383779e-01]
        Per-sample quantiles: [0.19345626 0.19713584 0.19912101 0.20286036 0.20737201 0.21262738
 0.21806726 0.22147903 0.22783792]

Histograms
        Bin Count:
                [ 1. 10. 21. 18. 11. 10. 10. 10. 15.  9.  5.  3.  2.  3.  4.  9.  5. 39. 69.  1.]
        Bin Centers:
                [1.3899655e-07 3.3902009e-07 8.0756939e-07 1.9051340e-06 4.4761473e-06
 1.0498669e-05 2.4606270e-05 5.7652869e-05 1.3506356e-04 3.1639601e-04
 7.4116164e-04 1.7361628e-03 4.0669274e-03 9.5266718e-03 2.2315973e-02
 5.2274540e-02 1.2245157e-01 2.8683940e-01 6.7191297e-01 1.2208900e+00]

Sharpness of unit activity:
        Low: [0.57754713 0.63841283 0.6745148  0.69920635]
        Middle: [0.3795108  0.28487584 0.22559366 0.18353757]
        High: [0.04294206 0.07671131 0.09989155 0.11725595]

Unused units:
        Threshold 0.001
          [42 48 55 58 63]
        Threshold 0.010
          [80 81 86 90 94]
        Threshold 0.050
          [108 111 114 119 120]
        Threshold 0.100
          [119 120 123 125 125]
        Threshold 0.150
          [124 125 125 125 126]

## CIFAR10

## CIFAR100