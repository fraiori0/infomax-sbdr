import jax
import jax.numpy as np
import flax.linen as nn
from typing import Dict, Any, Sequence

""" Utility functions and modules"""

def threshold_softgradient(x, threshold):
    """ Threshold function with the gradient of a sigmoid"""
    zero = jax.nn.sigmoid(x) - jax.lax.stop_gradient(jax.nn.sigmoid(x))
    return zero + (x >= threshold).astype(x.dtype)

def eligibility_step(e_prev, x, gamma):
    """ Update eligibility trace with decay and new input. """
    e_new = gamma * e_prev * (1 - x) + x
    return e_new

# Compute dimensions of convolutional output
def conv_output_dim(input_dim, kernel_size, stride, padding):
    if padding == 'SAME':
        return int(np.ceil(input_dim / stride))
    elif padding == 'VALID':
        return int(np.ceil((input_dim - kernel_size + 1) / stride))
    else:
        raise ValueError("Padding must be 'SAME' or 'VALID'")

class EMA(nn.Module):
    """ Layer similar to BatchNorm, but returns an exponential moving average of the input. """
    momentum: float = 0.95
    train: bool = True  # Whether in training mode
    axis: Sequence[int] = (-1,)  # Axes to keep when computing average
    @nn.compact
    def __call__(self, x):
        shape_features = [x.shape[ax] for ax in self.axis]

        ema = self.variable(
            "ema",
            "value",
            lambda: np.zeros(shape_features, dtype=x.dtype),
        )
        
        # Update EMA only if in training mode
        if self.train:
            # flatten x on all but the specified axes
            x = x.reshape((-1, *shape_features))
            # Compute average
            avg = np.mean(x, axis=0)
            # update ema only if it's not a call to init parameters (initialization)
            if not self.is_initializing():
                ema.value = self.momentum * ema.value + (1 - self.momentum) * avg

        # in any case, return the current value of the ema
        return ema.value


""" Spiking layers with eligibility traces """

class SpikeEligibilityDense(nn.Module):
    """
    Dense layer with spiking units and weights applied to eligibility traces of spikes.
    
    This layer have:
    - Forward weights (W_f) from input to unit (input is assumed to be an eligibility trace already)
    - Lateral weights (W_l) recurrent: from eligibility trace of other units to unit
    - An exponential moving average of the eligibility trace of the output units
    
    """
    
    n_units: int
    gamma_f: float = 0.9
    gamma_l: float = 0.9
    ema_momentum: float = 0.95
    p_target: float = 0.05  # Target average activation probability
    train: bool = True  # Whether in training mode
    threshold: float = None  # If set, use threshold function (with soft gradient) instead of sigmoid activation
    
    def setup(self):
        # forward weights
        self.forward = nn.Dense(
            features=self.n_units,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(0.0),
        )
        # lateral weights
        self.lateral = nn.Dense(
            features=self.n_units,
            kernel_init=nn.initializers.constant(0.0),
            use_bias=False,
            # bias_init=nn.initializers.constant(self.bias_init_value),
        )
    
        # # exponential-weighted moving average eligibility trace of the layer's own output
        # self.ema_e_l = EMA(
        #     momentum=self.ema_momentum,
        #     train=self.train,
        #     axis=(-1,),
        # )

    def __call__(self, e_x, e_l_prev):
        # Forward pass
        y_f = self.forward(e_x)
        # Lateral pass
        y_l = self.lateral(e_l_prev)
        # Combine and apply activation function
        y = y_f + y_l
        if self.threshold is not None:
            y = threshold_softgradient(y, self.threshold)
        else:
            y = jax.nn.sigmoid(y)

        # Update eligibility trace of output
        e_l = eligibility_step(e_l_prev, y, self.gamma_l)

        # # Update EMA of output eligibility trace
        # e_l_ema = self.ema_e_l(e_l)

        return {"out": y, "e_l": e_l,}# "e_l_ema": e_l_ema}
    
    def scan(self, e_x, e0_l):
        """
        Scan over time for the layer.
        
        Args:
            e_x: Input eligibility traces over time, shape (*batch_dims, time, input_dim)
            e0_l: Initial eligibility trace of the layer's output, shape (*batch_dims, n_units)
        
        Returns:
            Dictionary with all activations and values
        """
        def f_scan(carry, input):
            e_l_prev = carry
            e_x = input
            out = self(e_x, e_l_prev)
            e_l = out["e_l"]
            return e_l, out
        
        # move scan axis as first (required by jax.flax.scan)
        e_x = np.moveaxis(e_x, -2, 0)  # time
        # scan
        e_l_final, outs = jax.lax.scan(f_scan, e0_l, e_x)

        # move back scan axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
    
    @nn.nowrap
    def gen_initial_state(self, key, x):
        """
        Generate initial eligibility trace of the layer's output, given an input x.
        
        Args:
            x: Input data, shape (*batch_dims, input_dim)
        
        Returns:
            Initial eligibility trace of the layer's output, shape (*batch_dims, n_units)
        """
        return np.zeros((*x.shape[:-1], self.n_units))


class SpikeEligibilityConv(nn.Module):
    """
    Convolutional layer with spiking units and weights applied to eligibility traces of spikes.
    
    Each unit (i.e., each feature of the convolutional kernel) has:
    - Forward weights (W_f) from input to unit (input is assumed to be an eligibility trace already)
    - Lateral weights (W_l) recurrent: from eligibility trace of other units, to unit
    - An exponential moving average of the eligibility trace of the output units (kernel features)
    """

    n_units: int
    kernel_size_f: Sequence[int] = (3, 3)
    kernel_size_l: Sequence[int] = (3, 3)
    # note: the lateral should always be 'SAME', to match the feature map size
    padding_f: str = 'SAME'
    stride_f: Sequence[int] = (1, 1)
    gamma_f: float = 0.9
    gamma_l: float = 0.9
    ema_momentum: float = 0.95
    p_target: float = 0.05  # Target average activation probability
    train: bool = True  # Whether in training mode
    threshold = None  # If set, use threshold function (with soft gradient) instead of sigmoid activation
    
    def setup(self):
        # forward weights
        self.forward = nn.Conv(
            features=self.n_units,
            kernel_size=self.kernel_size_f,
            strides=self.stride_f,
            padding=self.padding_f,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(0.0),
        )
        # lateral weights
        self.lateral = nn.Conv(
            features=self.n_units,
            kernel_size=self.kernel_size_l,
            strides=(1, 1),
            padding='SAME',
            kernel_init=nn.initializers.constant(0.0),
            use_bias=False,
        )
    
        # exponential-weighted moving average eligibility trace of the layer's own output
        self.ema_e_l = EMA(
            momentum=self.ema_momentum,
            train=self.train,
            # Note, here we keep the spatial dimensions as well
            axis=(-3, -2, -1),
        )

    def __call__(self, e_x, e_l_prev):
        # Forward pass
        y_f = self.forward(e_x)
        # Lateral pass
        y_l = self.lateral(e_l_prev)
        # Combine and apply activation function
        y = y_f + y_l
        if self.threshold is not None:
            y = threshold_softgradient(y, self.threshold)
        else:
            y = jax.nn.sigmoid(y)

        # Update eligibility trace of output
        e_l = eligibility_step(e_l_prev, y, self.gamma_l)

        # Update EMA of output eligibility trace
        e_l_ema = self.ema_e_l(e_l)

        return {"out": y, "e_l": e_l, "e_l_ema": e_l_ema}
    
    def scan(self, e_x, e0_l):
        """
        Scan over time for the layer.
        
        Args:
            e_x: Input eligibility traces over time, shape (*batch_dims, time, height, width, input_dim)
            e0_l: Initial eligibility trace of the layer's output, shape (*batch_dims, height, width, n_units)
        
        Returns:        Dictionary with all activations and values
        """
        def f_scan(carry, input):
            e_l_prev = carry
            e_x = input
            out = self(e_x, e_l_prev)
            e_l = out["e_l"]
            return e_l, out
        
        # move scan axis as first (required by jax.flax.scan)
        e_x = np.moveaxis(e_x, -4, 0)  # time
        # scan
        e_l_final, outs = jax.lax.scan(f_scan, e0_l, e_x)

        # move back scan axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -4), outs)

        return outs
    
    @nn.nowrap
    def gen_initial_state(self, key, x):
        """
        Generate initial eligibility trace of the layer's output, given an input x.
        
        Args:
            x: Input data, shape (*batch_dims, height, width, input_dim)
        
        Returns:
            Initial eligibility trace of the layer's output, shape (*batch_dims, height, width, n_units)
        """
        # note, we need to compute the output shape after convolution
        # get input spatial dimensions
        input_height = x.shape[-3]
        input_width = x.shape[-2]
        # compute output spatial dimensions
        output_height = conv_output_dim(input_height, self.kernel_size_f[0], self.stride_f[0], self.padding_f)
        output_width = conv_output_dim(input_width, self.kernel_size_f[1], self.stride_f[1], self.padding_f)
        return np.zeros((*x.shape[:-3], output_height, output_width, self.n_units))

""" Modules with multiple spike eligibility layers """

class SpikeEligibilityModule(nn.Module):

    # parameters are given as a sequence, with one element per layer
    n_units: Sequence[int]
    gamma_f: Sequence[float] = 0.9
    gamma_l: Sequence[float] = 0.9
    ema_momentum: Sequence[float] = 0.95
    p_target: Sequence[float] = 0.05  # Target average activation probability
    train: bool = True  # Whether in training mode
    threshold: Sequence[float] = None  # If set, use threshold function (with soft gradient) instead of sigmoid activation

    def setup(self):
        layers = []
        for i in range(len(self.n_units)):
            layers.append(
                SpikeEligibilityDense(
                    n_units=self.n_units[i],
                    gamma_f=self.gamma_f[i],
                    gamma_l=self.gamma_l[i],
                    ema_momentum=self.ema_momentum[i],
                    p_target=self.p_target[i],
                    train=self.train,
                    threshold=self.threshold[i] if self.threshold is not None else None,
                )
            )
        self.layers = layers

    def __call__(self, e_x, *e_l):
        # x of shape (*batch_dims, input_dim)
        # e_l a list of eligibility traces for each layer, given as args of shape (*batch_dims, n_units_layer_i)
        outs = {}
        e_x_i = e_x
        for i in range(len(self.n_units)):
            e_l_i = e_l[i]
            out = self.layers[i](e_x_i, e_l_i)
            # Input to next layer is the eligibility trace of the output of this layer
            e_x_i = jax.lax.stop_gradient(out["e_l"])
            # append ouutput
            for k in out:
                if i == 0:
                    outs[k] = [out[k]]
                else:
                    outs[k].append(out[k])
        
        # convert outs to tuple
        # outs = {k: tuple(v) for k, v in outs.items()}

        return outs
    
    def scan(self, e_x, *e0_l):
        """
        Scan over time for the module.
        
        Args:
            e_x: Input eligibility traces over time, shape (*batch_dims, time, input_dim)
            e0_l: Initial eligibility traces of each layer's output, given as args of shape (*batch_dims, n_units_layer_i)
        
        Returns:
            Dictionary with all activations and values
        """
        
        def f_scan(carry, input):
            e_l_prev = carry
            e_x = input
            outs = self(e_x, *e_l_prev)
            e_l = outs["e_l"]
            return e_l, outs
        
        # move scan axis as first (required by jax.flax.scan)
        e_x = np.moveaxis(e_x, -2, 0)  # time
        # scan
        e_l_last, outs = jax.lax.scan(f_scan, list(e0_l), e_x)

        # move back scan axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs

    @nn.nowrap
    def gen_initial_state(self, key, x):
        """
        Generate initial eligibility traces of each layer's output, given an input x.
        
        Args:
            x: Input data, shape (*batch_dims, input_dim)
        
        Returns:
            Initial eligibility traces of each layer's output, shape (*batch_dims, n_units_layer_i)
        """
        batch_dims = x.shape[:-1]
        e0_l = []
        for n in self.n_units:
            key, _ = jax.random.split(key)
            e0_l.append(np.zeros((*batch_dims, n)))
        return e0_l