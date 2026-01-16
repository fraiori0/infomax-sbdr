import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc
from math import prod


""" Custom Pool functions"""


def or_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = 1.0 - inputs
    y = nn.pool(y, 1.0, jax.lax.mul, window_shape, strides, padding)
    return 1.0 - y


def and_pool(inputs, window_shape, strides=None, padding="VALID"):
    y = nn.pool(y, 1.0, jax.lax.mul, window_shape, strides, padding)
    return y


""" Convolutional modules """


class ConvBase(nn.Module):
    """
    Model with a sequence of convolutional layers, without pooling.
    """

    kernel_features: Sequence[int] = None
    kernel_sizes: Sequence[Tuple[int]] = None
    kernel_strides: Sequence[Tuple[int, int]] = None
    kernel_padding: str = "SAME"
    pool_fn: Callable = None
    pool_sizes: Sequence[Tuple[int, int]] = None
    pool_strides: Sequence[Tuple[int, int]] = None
    activation_fn: Callable = nn.leaky_relu

    def __call__(self, x):
        # input x of shape (*batch_dims, channel, height, width)
        x = self.layers(x)
        return x


class ConvNoPoolBase(ConvBase):
    """
    Model with a sequence of convolutional layers, without pooling.
    """

    def setup(self):
        assert len(self.kernel_features) == len(self.kernel_sizes)
        assert len(self.kernel_sizes) == len(self.kernel_strides)

        layers = []
        for f, k, s in zip(
            self.kernel_features, self.kernel_sizes, self.kernel_strides
        ):
            layers.append(
                nn.Conv(
                    features=f,
                    kernel_size=k,
                    strides=s,
                    padding=self.kernel_padding,
                ),
            )
            layers.append(self.activation_fn)
        # create the callable applying all layers in sequence
        self.layers = nn.Sequential(layers)


class ConvNoPoolNoLastBase(ConvBase):
    """
    Model with a sequence of convolutional layers, without pooling.
    Note, last layer does not have an activation
    """

    def setup(self):
        assert len(self.kernel_features) == len(self.kernel_sizes)
        assert len(self.kernel_sizes) == len(self.kernel_strides)

        layers = []
        for f, k, s in zip(
            self.kernel_features, self.kernel_sizes, self.kernel_strides
        ):
            layers.append(
                nn.Conv(
                    features=f,
                    kernel_size=k,
                    strides=s,
                    padding=self.kernel_padding,
                ),
            )
            layers.append(self.activation_fn)
        # remove last activation
        layers = layers[:-1]
        # create the callable applying all layers in sequence
        self.layers = nn.Sequential(layers)


class ConvFLONoPoolNoLast(ConvNoPoolNoLastBase):

    def setup(self):

        super().setup()

        # add a final layer returning the neg_pmi
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        x = self.layers(x)
        # flatten x
        x = x.reshape((*x.shape[:-3], -1))
        # sigmoid
        x = nn.sigmoid(x)
        # neg_pmi
        neg_pmi = self.neg_pmi_layer(x)
        return x, neg_pmi


class ConvFLONoPool(ConvNoPoolBase):
    output_features: int = None

    def setup(self):

        super().setup()
        # add a final dense layer
        self.final_dense = nn.Dense(features=self.output_features)
        # add a final layer returning the neg_pmi
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        x = self.layers(x)
        # flatten x
        x = x.reshape((*x.shape[:-3], -1))
        # apply final dense layer
        x = self.final_dense(x)
        # sigmoid
        x = nn.sigmoid(x)
        # neg_pmi
        neg_pmi = self.neg_pmi_layer(x)
        return x, neg_pmi


class VGGLayer(nn.Module):
    """A VGG-style module with two convolutional ;ayers followed by a pool-max."""

    kernel_features: int = None
    kernel_sizes: Tuple[int] = (3, 3)
    kernel_strides: Tuple[int, int] = (1, 1)
    kernel_padding: str = "SAME"
    pool_sizes: Tuple[int, int] = (2, 2)
    pool_strides: Tuple[int, int] = (2, 2)
    pool_padding: str = "SAME"
    activation_fn: Callable = nn.relu
    use_batchnorm: bool = False
    use_dropout: float = False
    dropout_rate: float = 0.2
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(
            features=self.kernel_features,
            kernel_size=self.kernel_sizes,
            strides=self.kernel_strides,
            padding=self.kernel_padding,
        )
        self.conv2 = nn.Conv(
            features=self.kernel_features,
            kernel_size=self.kernel_sizes,
            strides=self.kernel_strides,
            padding=self.kernel_padding,
        )
        self.pool = partial(
            nn.max_pool,
            window_shape=self.pool_sizes,
            strides=self.pool_strides,
            padding=self.pool_padding,
        )

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm(use_running_average=not self.training)
            self.bn2 = nn.BatchNorm(use_running_average=not self.training)

        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not self.training,
            )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.activation_fn(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = self.pool(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class VGGTransposeLayer(nn.Module):
    """A transpose VGG-style module with a conv-tranpose upsampling layer followed by a conv-tranpose layer."""

    kernel_features: int = None
    kernel_sizes: Tuple[int] = (3, 3)
    kernel_strides: Tuple[int, int] = (2, 2)
    kernel_padding: str = "SAME"
    # pool_sizes = None
    # pool_strides: Tuple[int, int] = (2, 2)
    # pool_padding = None
    activation_fn: Callable = nn.leaky_relu
    second_activation: bool = True
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = None
    training: bool = True
    

    def setup(self):
        # we don't use a separate upsampling layer, so the first
        # conv-transpose layer match the stride of the pooling layer
        # NOTE, this does not work if the stride of the kernel is not 1
        self.t_conv1 = nn.ConvTranspose(
            features=self.kernel_features,
            kernel_size=self.kernel_sizes,
            strides=self.kernel_strides,
            padding=self.kernel_padding,
        )
        self.t_conv2 = nn.ConvTranspose(
            features=self.kernel_features,
            kernel_size=self.kernel_sizes,
            strides=(1, 1),
            padding=self.kernel_padding,
        )

        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm(use_running_average=not self.training)
            self.bn2 = nn.BatchNorm(use_running_average=not self.training)

        if self.use_dropout:
            self.dropout = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=not self.training,
            )

    def __call__(self, x):
        # here we do dropout before the layers
        if self.use_dropout:
            x = self.dropout(x)
        x = self.t_conv1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.t_conv2(x)
        if self.second_activation:
            if self.use_batchnorm:
                x = self.bn2(x)
            x = self.activation_fn(x)
        return x


class VGG(nn.Module):
    out_features: int
    kernel_features: Sequence[int]
    activation_fn: Callable = nn.relu
    use_batchnorm: bool = False
    use_dropout: float = False
    dropout_rates: Sequence[float] = None
    training: bool = True
    out_activation_fn: Callable = nn.sigmoid

    def setup(self):
        conv_layers = []
        for f, dropout_rate in zip(self.kernel_features, self.dropout_rates):
            conv_layers.append(
                VGGLayer(
                    kernel_features=f,
                    activation_fn=self.activation_fn,
                    use_batchnorm=self.use_batchnorm,
                    use_dropout=self.use_dropout,
                    dropout_rate=dropout_rate,
                    training=self.training,
                ),
            )

        self.conv_layers = nn.Sequential(conv_layers)

        self.dense_layers = nn.Sequential(
            [
                nn.Dense(features=self.out_features),
                self.out_activation_fn,
            ]
        )

    def __call__(self, x):
        x = self.conv_layers(x)
        conv_shape = x.shape[-3:]
        # flatten the last three dimensions (i.e., C, H, W)
        x = x.reshape((*x.shape[:-3], -1))
        x = self.dense_layers(x)
        return {"z": x, "conv_shape": conv_shape}

class VGGDecoder(nn.Module):
    """Decoder architecture that can be used after a VGG architecture."""
    deconv_kernel_features: Sequence[int] = None
    preconv_chw: Sequence[int] = None
    activation_fn: Callable = nn.leaky_relu
    use_batchnorm: bool = False
    use_dropout: float = False
    dropout_rates: Sequence[float] = None
    training: bool = True

    def setup(self):
        in_features = prod(self.preconv_chw)

        self.decoder_dense = nn.Sequential(
            [
                nn.Dense(features=in_features),
                self.activation_fn,
            ]
        )

        conv_transpose_layers = []
        # the last layer has no activation function on the output
        for i in range(len(self.deconv_kernel_features)-1):
            conv_transpose_layers.append(
                VGGTransposeLayer(
                    kernel_features=self.deconv_kernel_features[i],
                    activation_fn=self.activation_fn,
                    use_batchnorm=self.use_batchnorm,
                    training=self.training,
                    second_activation=True,
                    use_dropout=self.use_dropout,
                    dropout_rate=self.dropout_rates[i],
                ),
            )
        conv_transpose_layers.append(
            VGGTransposeLayer(
                kernel_features=self.deconv_kernel_features[-1],
                activation_fn=self.activation_fn,
                use_batchnorm=self.use_batchnorm,
                training=self.training,
                second_activation=False,
                use_dropout=self.use_dropout,
                dropout_rate=self.dropout_rates[-1],
            ),
        )

        self.conv_transpose_layers = nn.Sequential(conv_transpose_layers)

    def __call__(self, x):
        x = self.decoder_dense(x)
        # reshape back to (*batch_dims, C, H, W)
        x = x.reshape((*x.shape[:-1], *self.preconv_chw))
        x = self.conv_transpose_layers(x)
        return {"z": x}


class VGGAutoEncoder(VGG):
    deconv_in_features: int = None
    deconv_kernel_features: Sequence[int] = None

    def setup(self):
        super().setup()

        self.decoder_dense = nn.Sequential(
            [
                nn.Dense(features=self.deconv_in_features),
                self.activation_fn,
            ]
        )

        conv_transpose_layers = []
        for f in self.deconv_kernel_features:
            conv_transpose_layers.append(
                VGGTransposeLayer(
                    kernel_features=f,
                    activation_fn=self.activation_fn,
                    use_batchnorm=self.use_batchnorm,
                    training=self.training,
                ),
            )

        self.conv_transpose_layers = nn.Sequential(conv_transpose_layers)

    def __call__(self, x):
        # # Encode
        x = self.conv_layers(x)
        # flatten the last three dimensions (i.e., C, H, W)
        x_f_shape = x.shape[-3:]
        x = x.reshape((*x.shape[:-3], -1))
        x = self.dense_layers(x)

        # # Decode
        y = self.decoder_dense(x)
        # reshape back to (*batch_dims, C, H, W)
        y = y.reshape((*y.shape[:-1], *x_f_shape))
        # convolutional transpose layers
        y = self.conv_transpose_layers(y)
        return {"z": x, "x_rec": y}

    def encode(self, x):
        x = self.conv_layers(x)
        # flatten the last three dimensions (i.e., C, H, W)
        x = x.reshape((*x.shape[:-3], -1))
        x = self.dense_layers(x)
        return {"z": x}


class VGGFLO(VGG):

    def setup(self):
        super().setup()
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        outs = super().__call__(x)
        neg_pmi = self.neg_pmi_layer(outs["z"])
        outs["neg_pmi"] = neg_pmi
        return outs
    
class VGGFLOMultiLayerNEGPMI(VGG):

    neg_pmi_hid_features: Sequence[int] = None

    def setup(self):
        super().setup()
        neg_pmi_layers = []
        for f in self.neg_pmi_hid_features:
            neg_pmi_layers.append(nn.Dense(features=f))
            neg_pmi_layers.append(self.activation_fn)
        neg_pmi_layers.append(nn.Dense(features=1))
        self.neg_pmi_layers = nn.Sequential(neg_pmi_layers)
        

    def __call__(self, x):
        outs = super().__call__(x)
        neg_pmi = self.neg_pmi_layers(outs["z"])
        outs["neg_pmi"] = neg_pmi
        return outs


class VGGFLOAutoEncoder(VGGAutoEncoder):

    def setup(self):
        super().setup()
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        outs = super().__call__(x)
        outs["neg_pmi"] = self.neg_pmi_layer(outs["z"])
        return outs


class VGGFLOKSoftMax(VGG):
    k_softmax: int = None

    def setup(self):
        assert self.out_features % self.k_softmax == 0
        self.k_shape = (self.k_softmax, self.out_features // self.k_softmax)

        super().setup()

        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        outs = super().__call__(x)
        z = outs["z"]
        # reshape
        z = z.reshape((*z.shape[:-1], *self.k_shape))
        # apply softmax
        z = jax.nn.softmax(z, axis=-1)
        # flatten
        z = z.reshape((*z.shape[:-2], -1))
        # compute neg pmi
        neg_pmi = self.neg_pmi_layer(z)
        return {"z": z, "neg_pmi": neg_pmi}


class VGGGlobalPool(nn.Module):
    out_features: int
    kernel_features: Sequence[int]
    activation_fn: Callable = nn.relu
    use_batchnorm: bool = False
    use_dropout: float = False
    dropout_rates: Sequence[float] = None
    training: bool = True
    out_activation_fn: Callable = nn.sigmoid

    def setup(self):
        conv_layers = []
        for f, dropout_rate in zip(self.kernel_features, self.dropout_rates):
            conv_layers.append(
                VGGLayer(
                    kernel_features=f,
                    activation_fn=self.activation_fn,
                    use_batchnorm=self.use_batchnorm,
                    use_dropout=self.use_dropout,
                    dropout_rate=dropout_rate,
                    training=self.training,
                ),
            )

        self.conv_layers = nn.Sequential(conv_layers)

        self.dense_layers = nn.Sequential(
            [   
                # nn.Dense(features=self.out_features),
                # self.activation_fn,
                nn.Dense(features=self.out_features),
                self.out_activation_fn,
            ]
        )

    def __call__(self, x):
        x = self.conv_layers(x)
        conv_shape = x.shape[-3:]
       # Perform Global Pooling
        x = np.mean(x, axis=(-3, -2))
        # x = np.max(x, axis=(-3, -2))
        x = self.dense_layers(x)
        return {"z": x, "conv_shape": conv_shape}


class VGGGlobalPoolFLO(VGGGlobalPool):

    def setup(self):
        super().setup()
        self.neg_pmi_layer = nn.Dense(features=1)

    def __call__(self, x):
        outs = super().__call__(x)
        outs["neg_pmi"] = self.neg_pmi_layer(outs["z"])
        return outs

class VGGGlobalPoolFLOMultiLayerNEGPMI(VGGGlobalPool):

    neg_pmi_hid_features: Sequence[int] = None

    def setup(self):
        super().setup()
        neg_pmi_layers = []
        for f in self.neg_pmi_hid_features:
            neg_pmi_layers.append(nn.Dense(features=f))
            neg_pmi_layers.append(self.activation_fn)
        neg_pmi_layers.append(nn.Dense(features=1))
        self.neg_pmi_layers = nn.Sequential(neg_pmi_layers)
        

    def __call__(self, x):
        outs = super().__call__(x)
        neg_pmi = self.neg_pmi_layers(outs["z"])
        outs["neg_pmi"] = neg_pmi
        return outs