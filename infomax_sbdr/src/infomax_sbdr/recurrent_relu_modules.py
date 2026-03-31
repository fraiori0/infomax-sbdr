import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.binary_comparisons as bc

""" Convolutional modules """

class RecReluLayer(nn.Module):
    """
    Module setup similarly to Recurrent Predictive Learning
    1. Feedforward only encoder
    2. Recurrent Elman Network
    3. Prediction Layer

    Meant to be trained with:
    - I(encoded input, prediction_prev) (maximize predictive InfoNCE)
    - I(recurrent state, recurrent state) (maximize contrastive entropy of recurrent state)
    """
    features: int

    def setup(self):
        self.fwd_layer = nn.Dense(self.features)
        self.rec_layer_input = nn.Dense(self.features, use_bias=False)
        self.rec_layer_lateral = nn.Dense(self.features, use_bias=False)
        self.pred_layer = nn.Dense(self.features)

    def __call__(self, x, state):
        # assume input x is of shape 
        # (*batch_dims, features), i.e., a single time-step
        # we will then scan this function over time

        # # Forward-only encoder
        a_fwd = jax.nn.tanh(self.fwd_layer(x))
        # binary activation
        z_fwd = (a_fwd > 0).astype(np.float32)
        
        # # Recurrent module
        a_rec = np.tanh(self.rec_layer_input(a_fwd*z_fwd) + self.rec_layer_lateral(state["a_rec"] * state["z_rec"]))
        # binary activation
        z_rec = (a_rec > 0).astype(np.float32)

        # # Prediction head
        a_pred = jax.nn.tanh(self.pred_layer(a_rec * z_rec))
        # binary activation
        z_pred = (a_pred > 0).astype(np.float32)

        outs = {
            "a_fwd": a_fwd,
            "z_fwd": z_fwd,
            "a_rec": a_rec,
            "z_rec": z_rec,
            "a_pred": a_pred,
            "z_pred": z_pred
        }

        new_state = {
            "a_rec": a_rec,
            "z_rec": z_rec
        }

        return new_state, outs
    
    def init_state_from_input(self, key, x):
        # assume input x is of shape 
        # (*batch_dims, features), i.e., a single time-step
        # we will then scan this function over time
        
        # init a random recurrent state a
        a = 0.1 * jax.random.normal(key, shape=(*x.shape[:-1], self.features)).astype(np.float32)
        # init z from a
        z = (a > 0).astype(np.float32)

        return {
            "a_rec": a,
            "z_rec": z
        }
    
    def scan(self, x_seq, state):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(x, state)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
        
    

class RecRelu(nn.Module):
    """
    A stack of RecReluLayers, with gradient blocking in the middle
    """
    features: Sequence[int]

    def setup(self):
        self.layers = [
            RecReluLayer(features) for features in self.features
        ]

    @nn.nowrap
    def init_state_from_input(self, key, x):
        states = []
        # note, we cannot call init_state_from_input for each layer, as we also need to do this 
        # before initialization of the module, to call init on the nn.Module
        # at that time, self.layers is not accessible, unless we use apply
        batch_shape = x.shape[:-1]
        for f in self.features:
            # init a random recurrent state a
            key, _ = jax.random.split(key)
            a = 0.1 * jax.random.normal(key, shape=(*batch_shape, f)).astype(np.float32)
            # init z from a
            z = (a > 0).astype(np.float32)
            states.append({
                "a_rec": a,
                "z_rec": z
            })

        return states
    
    def __call__(self, x, states):

        new_states, outs = [], []

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(x, states[l_idx])
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x = jax.lax.stop_gradient(out["a_rec"]*out["z_rec"])

        return new_states, outs
    
    def scan(self, x_seq, states):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, layer in enumerate(self.layers):
            # note, scan does not return the final state
            out = layer.scan(x_seq, states[l_idx])
            outs.append(out)

            # set input sequence for the next layer
            x_seq = jax.lax.stop_gradient(out["a_rec"]*out["z_rec"])

        return outs