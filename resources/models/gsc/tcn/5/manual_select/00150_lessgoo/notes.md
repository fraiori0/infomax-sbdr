# Lesss goooo?

Quite impressive results on GSC (40 log-mel features, CMVN normalization, standard params of standard GSCDataset)

95.5% accuracy (on validation set, 0.16 split)

This folder contains also the config file.

## Optimizer

Using a lower learning rate (and weight decay) works to stabilize and largely avoid overfitting
```toml
type = "adamw"
[training.optimizer.kwargs]
learning_rate = 0.0001
weight_decay = 0.000001
```

## Class

```python
class TemporalConvPoolLayer(nn.Module):
    """Causal temporal convolutional layer.
 
    Attributes:
        features:      Number of output channels (kernel features).
        kernel_size:   Length of the 1-D convolutional kernel (must be ≥ 1).
        kernel_stride: Step size along the time axis (default 1).
        pool_size:     Length of the 1-D pooling kernel (must be ≥ 1).
        pool_stride:   Step size along the time axis (default 1).
        use_bias:      Whether to add a learnable bias term.
        kernel_init:   Initialiser for the convolutional kernel weights.
        bias_init:     Initialiser for the bias vector.
    """
 
    features: int
    kernel_size: int
    kernel_stride: int
    pool_size: int
    pool_stride: int
    use_bias: bool = True
 
    def setup(self) -> None:
        if self.kernel_size < 1:
            raise ValueError(f"kernel_size must be ≥ 1, got {self.kernel_size}")
        if self.kernel_stride < 1:
            raise ValueError(f"stride must be ≥ 1, got {self.kernel_stride}")
 
        # We manage causal padding ourselves, so the underlying Conv uses
        # 'VALID' (no implicit padding).
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.kernel_stride,),
            padding="VALID",
            use_bias=self.use_bias,
            bias_init=nn.initializers.constant(0.5),
        )
 
    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal temporal convolution.
 
        Args:
            x: Float array of shape ``(*batch_dims, time, features)``.
               Any number of leading batch dimensions is supported.
 
        Returns:
            Float array of shape ``(*batch_dims, time_out, self.features)``
            where ``time_out = ⌊(time − 1) / stride⌋ + 1``.
        """
        # 1. Causal padding
        # Prepend (kernel_size − 1) zero frames so each output step only
        # sees the present and past inputs, never future ones.
        
        # np.pad expects a sequence of (before, after) pairs, one per axis.
        # Layout: [...batch axes..., time axis, feature axis]
        pad_size = self.kernel_size - 1
        if pad_size > 0:
            pad_width = [(0, 0)] * (x.ndim - 2) + [(pad_size, 0), (0, 0)]
            x = np.pad(x, pad_width)
 
        # 2. Convolution (VALID — no further padding)
        pre_activation = self.conv(x)
        # z = jax.nn.sigmoid(pre_activation)
        # y = ut.threshold_softgradient(pre_activation)
        z = directional_clip(pre_activation, lo=0.0, hi=1.0)
        y = np.where(z > 0.1, 1.0, 0.0)


        # 3. Aggregate temporally using max_pool or avg_pool
        p = nn.max_pool(
            z,
            window_shape=(self.pool_size,),
            strides=(self.pool_stride,),
            padding="SAME",
        )
        # p=z
        # # Alternative: a 1/t kernel
        # w_pool = 1.0/np.arange(1, self.pool_size+1)
        # # collapse batch dimensions
        # p = ut.strided_time_conv(z, w_pool, self.pool_stride)

        outs = {
            "z": z,
            "y": y,
            "p": p,
        }
        return outs

class TCNPoolClassifier(nn.Module):
    """Temporal Convolutional Network (TCN) with multiple layers and pooling after convolutions.
    And a final linear layer for classification using outer product of 
    sparse binary activations from the last two layers.
    """
 
    features: Sequence[int]
    kernel_sizes: Sequence[int]
    kernel_strides: Sequence[int]
    pool_sizes: Sequence[int]
    pool_strides: Sequence[int]
    class_features: int
    use_bias: bool = True
    stop_grad: bool = False
    stop_grad_class: bool = False
    binarize: bool = False
 
    def setup(self) -> None:
        if len(self.features) != len(self.kernel_sizes):
            raise ValueError("features and kernel_sizes must have the same length")
        if isinstance(self.kernel_strides, int):
            self.kernel_strides = [self.strides] * len(self.features)
        elif len(self.kernel_strides) != len(self.features):
            raise ValueError("strides must be an int or have the same length as features")
 
        self.layers = [
            TemporalConvPoolLayer(
                features=self.features[i],
                kernel_size=self.kernel_sizes[i],
                kernel_stride=self.kernel_strides[i],
                pool_size=self.pool_sizes[i],
                pool_stride=self.pool_strides[i],
                use_bias=self.use_bias,
            )
            for i in range(len(self.features))
        ]

        # classifier, the extra feature is treated as a gate on the logits
        self.classifier = nn.Dense(self.class_features + 1) 
    
    def out_to_next(self, out):
        x = out["p"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        if self.binarize:
            x = np.where(x > 0.1, x, 0.0)
        return x

    def gather_input_classifier(self, outs):
        x = outs[-1]["p"]
        if self.stop_grad_class:
            x = jax.lax.stop_gradient(x)
        if self.binarize:
            x = np.where(x > 0.1, x, 0.0)
        return x
    
    def class_out_to_logits(self, class_out):
        gate = jax.nn.sigmoid(class_out[..., -1])
        logits = class_out[..., :-1] * gate[..., None]
        return logits

    def __call__(self, x):
        # # # Forward psss
        # return all intermediate outputs
        outs = []
        x_in = x
        for layer in self.layers:

            out = layer(x_in)
            outs.append(out)
            
            # Input to next layer
            x_in = self.out_to_next(out)
    
        # Compute linear layer
        class_in = self.gather_input_classifier(outs)
        class_out = self.classifier(class_in)
        logit = self.class_out_to_logits(class_out)

        aux = {
            "logit": logit,
        }

        return outs, aux
```

## Config

```toml
[model]
type = "TCNPoolClassifier"
seed = 46
p_target = 0.01
[model.kwargs]
features = [256, 256, 256]
kernel_sizes = [3, 3, 3]
kernel_strides = [1, 1, 1]
pool_sizes = [3, 3, 3]
pool_strides = [2, 2, 2]
class_features = 35 # number of classes
stop_grad = false
stop_grad_class = false
binarize = true

[dataset]
[dataset.transform.specaugment.kwargs]
n_time_masks = 2
max_time_width = 5 # over 98 steps, with standard Mel-Spectrogram params on GSC dataset
n_freq_masks = 2
max_freq_width = 2

[training]
epochs = 150
batch_size = 256
num_classes = 35

[training.checkpoint]
save = true
save_interval = 2 # epochs
max_to_keep = 12

[training.dataloader]
shuffle = true
# drop the last batch to avoid recompiling jitted functions because of changing input dims
# see here https://flax-linen.readthedocs.io/en/latest/guides/data_preprocessing/full_eval.html
drop_last = true

[training.loss]
eps = 1.0e-1
w_eps = 1.0e-2
class_steps = 50

[training.optimizer]
type = "adamw"
[training.optimizer.kwargs]
learning_rate = 0.0001
weight_decay = 0.000001
# momentum=0.9
# nesterov=true

[validation]
split = 0.16
eval_interval = 12
[validation.dataloader]
batch_size = 256
[validation.sim_fn]
type = "log_and"
quantile = 0.9
[validation.sim_fn.kwargs]
eps = 1.0e-2
```

## Details


1) Dataset: Google Speech Commands V2 (35 classes). The dataset is preprocessed in the "standard" way. I extract 40 features log-mel spectrograms (minimum frequency 20Hz, maximum frequency 8000, padding to 1 second, 98 steps for each sample) and normalize per-utterance with cepstral mean and variance. Results below are reported for the validation set, I haven't used the test set yet (to keep model selection, and in general reasoning about the model, clean).
2) Accuracy: 95.7% (on validation set). Interestingly, I get a 0.9999 accuracy on the training set, somehow this model is a super-strong overfitter (but as long as I do not impose too much sparsity, like towards 1%, and keep a low learning rate, it learns also well for unseen samples in the validation set, i.e., I tend to believe the accuracy is good).
3) Architecture: the architecture is a simple Temporal Convolutional Network (with causal padding), followed by a single layer linear classifier (emitting logits over the 35 classes).
3.a) TCN: three standard 1d convolutional layers (I did not use "blocks", batch normalization or other composed architecture, just single layers) with non-linear bounded activation ($\in [0,1]$). Every layer has 256 output unit, a kernel size of 3, a kernel stride of 1, followed by non-linear activation and max-pooling with size 3 and stride 2.
3.b) In the architecture, I did a "trick" with the activation function, after multiple test on different ideas. I used a clip(x, 0,1) activation function, but with a custom gradient, that I attach below (as code) as it is easier than explaining it. This made a quite more powerful model than using sigmoid activation (in terms of fast fitting/maximum performance, but also more prone to overfitting under too big learning rate and too high sparsity). Sigmoid activated networks can also learn pretty well. Also, as mentioned, before passing the activation to the next layer I threshold them, setting exactly to 0 everything lower than 0.1, guaranteeing that we have hard sparsity. This avoids that this low-activated units receive gradients from above, helping the model to achieve proper sparsity. The InfoNCE regularization loss is applied to the output of each layer (before thresholding, so it can push for actual zero or "resurrect" dead units). 
3.b) Classifier: the classifier is a single linear layer, taking as input the output of the last 1d convolutional layer, but with one extra output unit (with respect to the number of classes) which is instead first passed through a sigmoid (only this unit) and treated as a gate. Each step from the last layer outputs logits. Classification is done by summing the logits (each multiplied by its gate value, so the model can "not use" uninformative steps), to get a single logit vector for the entire sequence.
4) Sparsity: first, let me explain what I meant by "zero-hard" sparse. By that, I mean that the statistics reported below are for the number of non-zero units. In my model, units below a certain activation threshold are zeroed out, such that are exact zero for the next layer. This avoids conflating a low but frequent average activation value vs actually being zero exactly, which is what matters for sparse computation (e.g., I can pass a value of 0.2 to the next layer, units are zeroed conditionally but not binarized). 
4.a) Per-Unit sparsity: below, I report quantiles of non-zero value. Per unit, this means the quantiles of the averages (for each unit, over samples) of the number of times a unit is non-zero. This show how the non-zero activity is distributed over units, i.e., whether units are activated homogeneously or not.
Quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
Layer 1: [0.0741, 0.0839, 0.0929, 0.1056, 0.1572]
Layer 2: [0.0357, 0.0415, 0.0461, 0.0517, 0.0649]
Layer 3: [0.0359, 0.0402, 0.0442, 0.0484, 0.0538]
4.b) Per-Sample sparsity: same as above, again counting non-zero values and not raw activation values, but the quantiles are for the average number of non-zero units in each sample.
Quantiles: [0.05, 0.25, 0.5, 0.75, 0.95]
Layer 1: [0.0155, 0.0586, 0.0974, 0.1382, 0.202]
Layer 2: [0.0191, 0.0312, 0.043, 0.0618, 0.0891]
Layer 3: [0.0156, 0.0253, 0.0391, 0.0553, 0.0905]
