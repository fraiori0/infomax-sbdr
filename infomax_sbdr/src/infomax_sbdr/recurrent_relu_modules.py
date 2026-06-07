import jax
import jax.numpy as np
from jax import jit, grad, vmap
import flax.linen as nn
from typing import Sequence, Callable, Tuple
from functools import partial
import infomax_sbdr.utils as ut
import infomax_sbdr.initializers as my_inits

""" Custom RPL-style module """

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
        # Forward
        self.fwd_layer = nn.Dense(self.features)
        # Recurrent
        self.rec_layer_input = nn.Dense(self.features, use_bias=True)
        self.rec_layer_lateral = nn.Dense(self.features, use_bias=False)
        # Prediction
        self.pred_layer = nn.Dense(self.features)

    def __call__(self, x, state):
        # assume input x is of shape 
        # (*batch_dims, features), i.e., a single time-step
        # we will then scan this function over time

        # # Forward-only encoder
        a_fwd = self.fwd_layer(x)
        a_fwd = ut.symlog(a_fwd)
        z_fwd = (a_fwd > 0).astype(np.float32)
        
        # # Recurrent module
        a_rec = (
            # Normal
            # self.rec_layer_input(z_fwd*a_fwd)
            # Stop gradient
            self.rec_layer_input(jax.lax.stop_gradient(z_fwd*a_fwd))
            + self.rec_layer_lateral(state["z_rec"]*state["a_rec"])
        )
        a_rec = ut.symlog(a_rec)
        z_rec = (a_rec > 0).astype(np.float32)

        # # Prediction head
        a_pred = self.pred_layer(z_rec*a_rec)
        a_pred = ut.symlog(a_pred)
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
    

""" Custom Vanilla Recurrent Layer """

class RecSTELayer(nn.Module):
    """
    """
    in_features: int
    out_features: int

    def setup(self):
        # # # Network weights
        # Note, we consider only non-negative weights
        
        # Forward (excitatory) weights
        self.w = self.param(
            "w",
            init_fn = nn.initializers.lecun_normal(),#my_inits.non_negative(scale=1.0/np.sqrt(self.in_features)),
            shape = (self.in_features, self.out_features),
            dtype = np.float32,
        )

        # Bias (homeostatic)
        self.b = self.param(
            "b",
            nn.initializers.constant(0.0),
            (self.out_features,),
            np.float32
        )

        # Average  binary activation and average memory state
        self.mem = nn.Dense(
            self.out_features,
            use_bias=True,
            kernel_init=nn.initializers.lecun_normal(),
            bias_init=nn.initializers.constant(0.0)
        )

    def __call__(self, state, x):
        # assume input x is of shape 
        # (*batch_dims, features), i.e., a single time-step
        # we will then scan this function over a time axis in another method, using jax.lax.scan
        
        th = 0.3
        gamma = 0.8

        # Compute pre-activation
        # a = ut.symlog(x @ self.w + self.b)
        # a = ut.threshold_softgradient(x @ self.w + self.b)
        a = jax.nn.sigmoid(x @ self.w + self.b)

        # # Compute binary activation leaving linear gradient only on b
        # zero_b = self.b - jax.lax.stop_gradient(self.b)
        # z = zero_b + jax.lax.stop_gradient(a + self.b > 0).astype(np.float32)
        z = (a > th).astype(np.float32)

        # Update memory state using InfoNCE-like single step update
        z_in = jax.lax.stop_gradient(z)
        s_prev = state["s"] # jax.lax.stop_gradient(state["s"]) # 

        r_in = np.concatenate((z_in, s_prev), axis=-1)
        s = jax.nn.sigmoid(self.mem(r_in))
        zs = (s > th).astype(np.float32)
        # zs = s

        outs = {
            "a": a,
            "z": z,
            "s": s,
            "zs": zs,
        }

        new_state = {
            "s": s,
        }

        return new_state, outs
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method
        
        batch_dims = x.shape[:-1]
        s = np.zeros(shape=(*batch_dims, self.out_features), dtype=np.float32)
        # eligibility traces for both input and state
        e_z = np.zeros(shape=(*batch_dims, self.out_features), dtype=np.float32)
        e_s = np.zeros(shape=(*batch_dims, self.out_features), dtype=np.float32)

        return {
            "s": s,
            "e_z": e_z,
            "e_s": e_s,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in first position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
        
    

class RecSTE(nn.Module):
    """
    A stack of RecSTELayers, with gradient blocking in the middle
    """
    in_features: int
    features: Sequence[int]

    def setup(self):
        layers = []
        f_in = self.in_features
        for f_out in self.features:
            layers.append(
                RecSTELayer(f_in, f_out)
            )
            f_in = f_out

        self.layers = layers

    @nn.nowrap
    def init_state_from_input(self, key, x):
        states = []
        # note, we cannot call init_state_from_input for each layer, as we also need to do this 
        # before initialization of the module, to call init on the nn.Module
        # at that time, self.layers is not accessible, unless we use apply
        batch_shape = x.shape[:-1]
        for f in self.features:
            # init a recurrent state s
            s= np.zeros((*batch_shape, f), dtype=np.float32)
            states.append({
                "s": s,
            })

        return states
    
    def __call__(self, states, x):

        new_states, outs = [], []

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x)
            new_states.append(st)
            outs.append(out)

            # set input to next layer (stop gradient)
            x = jax.lax.stop_gradient(out["a"])

        return new_states, outs
    
    def scan(self, states, x_seq):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_seq)
            outs.append(out)

            # set input sequence for the next layer
            x_seq = jax.lax.stop_gradient(out["a"])

        return outs
    

class RecSTEClassifier(RecSTE):
    """
    A stack of RecSTELayers, with gradient blocking in the middle, and a final single layer classifier
    """
    out_labels: int
    classf_layer_idx: int = 0

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    def gather_input_classifier(self, outs):
        return jax.lax.stop_gradient(outs[self.classf_layer_idx]["s"])
    
    def __call__(self, states, x):

        new_states, outs = super().__call__(states, x)

        # compute logits
        logits = self.classifier(self.gather_input_classifier(outs))
        
        aux = {
            "logit": logits,
        }

        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_seq)
            outs.append(out)

            # set input sequence for the next layer
            x_seq = jax.lax.stop_gradient(out["s"])

        # compute logits
        logits = self.classifier(self.gather_input_classifier(outs))
        
        aux = {
            "logit": logits,
        }

        return outs, aux

class SparseGRULayer(nn.Module):
    """
    GRU-like implementation, supposed to be used with sparse activations
    """

    features: int

    def setup(self):
        self.forward = nn.Dense(self.features)
        self.reset = nn.Dense(self.features)

    def __call__(self, state, x):
        # Single time step, x assumed of shape (*, features)

        z_prev = state["z"]

        r = jax.nn.tanh(self.reset(np.concatenate((x, z_prev), axis=-1)))
        h = jax.nn.sigmoid(self.forward(np.concatenate((x, r*z_prev), axis=-1)))
        
        # softXOR to compute the new state
        z = h + z_prev - 2 * h * z_prev

        out = {
            "h": h,
            "z": z,
        }

        state = {
            "z": z,
        }

        return state, out
    
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method, i.e., before initialization
        
        batch_dims = x.shape[:-1]
        z = np.zeros(shape=(*batch_dims, self.features), dtype=np.float32)

        return {
            "z": z,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs

class SparseGRU(nn.Module):
    """
    A stack of SparseGRULayers
    """
    features: Sequence[int]
    stop_grad: bool

    def setup(self):
        self.layers = [SparseGRULayer(f) for f in self.features]
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method
        states = []
        batch_shape = x.shape[:-1]
        for f in self.features:
            # init a recurrent state s
            z= np.zeros((*batch_shape, f), dtype=np.float32)
            states.append({
                "z": z,
            })

        return states
    
    def out_to_next(self, out):
        # given the output of a single layer, create the input to the next layer
        # Here is the single point to control whether we have or not gradient flow
        x = out["z"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)

        return x
    
    def __call__(self, states, x):

        new_states, outs = [], []

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x)
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x = self.out_to_next(out)

        return new_states, outs
    
    def scan(self, states, x_seq):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_seq)
            outs.append(out)

            # set input sequence for the next layer
            x_seq = self.out_to_next(out)

        return outs
    

class SparseGRUClassifier(SparseGRU):
    """
    A stack of SparseGRULayers with a final classifier head
    """
    out_labels: int

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    
    def gather_input_classifier(self, outs):
        return outs[-1]["z"]
    
    def __call__(self, states, x):
        new_states, outs = super().__call__(states, x)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):
        outs = super().scan(states, x_seq)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return outs, aux
    
class CloneStructureLayer(nn.Module):
    """
    A recurrent module with cloned-input hidden units, similarly to 
    Hierarchical Temporal Memory or Clone Structure Cognitive Graphs
    """

    features : int
    clones : int

    def setup(self):

        self.forward = nn.Dense(self.features)
        self.recurrent = nn.Dense(self.features * self.clones)

    def __call__(self, state, x):
        # Single time step, x assumed of shape (*, features)

        z_prev = state["z"]
        # # unflatten
        # z_prev = z_prev.reshape((*z_prev.shape[:-1], self.features, self.clones))

        # Compute forward activation
        y_f = jax.nn.sigmoid(self.forward(x))
        # # Compute recurrent activation
        z_l = jax.nn.sigmoid(self.recurrent(z_prev))

        # # Reshape z_l 
        z_l = z_l.reshape((*z_l.shape[:-1], self.features, self.clones))
        
        # Alternative with same weights for each row of clones

        # Compute gating. All clones receive the same forward input, 
        # but a different lateral input 
        # AND
        z = y_f[..., None] * z_l

        # Clone-pool version (before flattening, we pool on the clones)
        y = z.max(-1)
        # Flatten features and clones
        z = z.reshape((*z.shape[:-2], -1))

        out = {
            "z": z,
            "y": y,
            "y_f": y_f,
        }
        
        state = {
            "z": z,
        }

        return state, out
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method, i.e., before initialization
        
        batch_dims = x.shape[:-1]
        z = np.zeros(shape=(*batch_dims, self.features*self.clones), dtype=np.float32)

        return {
            "z": z,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs

class CloneStructure(nn.Module):
    """
    A stack of CloneStructuredLayers
    """
    features: Sequence[int]
    clones: Sequence[int]
    stop_grad: bool = False

    def setup(self):
        self.layers = [
            CloneStructureLayer(features=f, clones=c)
            for f, c in zip(self.features, self.clones)
        ]

    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state
        # Note: this method can be called without the "apply" method
        states = []
        batch_shape = x.shape[:-1]
        for f, c in zip(self.features, self.clones):
            # init a recurrent state s
            z= np.zeros((*batch_shape, f*c), dtype=np.float32)
            states.append({
                "z": z,
            })

        return states

    def out_to_next(self, out):
        x = out["z"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        return x

    def __call__(self, states, x):

        new_states, outs = [], []

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x)
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x = self.out_to_next(out)

        return new_states, outs
    
    def scan(self, states, x_seq):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_seq)
            outs.append(out)

            # set input sequence for the next layer
            x_seq = self.out_to_next(out)

        return outs
    

class CloneStructureClassifier(CloneStructure):
    """
    A stack of CloneStructure layers with a classifier head
    """
    out_labels: int = None

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    
    def gather_input_classifier(self, outs):
        return outs[-1]["z"]
    
    def __call__(self, states, x):
        new_states, outs = super().__call__(states, x)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):
        outs = super().scan(states, x_seq)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return outs, aux
    
class TransformerLayer(nn.Module):
    """
    A single transformer layer with recurrent attention
    """

    qk_features: int
    v_features: int

    def setup(self):
        self.q_proj = nn.Dense(
            self.qk_features,
            bias_init=nn.initializers.constant(-1.0),
        )
        self.k_proj = nn.Dense(
            self.qk_features,
            bias_init=nn.initializers.constant(-1.0),
        )
        self.v_proj = nn.Dense(self.v_features)
        self.gate = nn.Dense(1)

    def __call__(self, state, x):

        # s of shape (*batch_dims, qk_features, v_features)
        s = state["s"]

        k = jax.nn.sigmoid(self.k_proj(x))
        q = jax.nn.sigmoid(self.q_proj(x))
        v = self.v_proj(x)
        # v = jax.nn.sparse_plus(self.v_proj(x))

        # read operation, o of shape (*batch_dims, v_features)
        # o = (q[..., :, None] * s).sum(-2)
        q_norm_sq = (q*q).sum(-1, keepdims=True) + 1e-8
        o = np.einsum("...q,...qv->...v", q/q_norm_sq, s)
        o = jax.nn.leaky_relu(o)

        # # state update using delta rule
        k_norm_sq = (k*k).sum(-1, keepdims=True) + 1e-8
        e = v - np.einsum("...k,...kv->...v", k, s)
        delta = np.einsum("...k,...v->...kv", k / k_norm_sq, e)
        s_new = s + delta
        # # Simply accumulate
        # # k_state = jax.lax.stop_gradient(k)
        # k_state = k
        # s_new = np.einsum("...k,...kv->...kv", (1-k_state), s) + np.einsum("...k,...v->...kv", k_state, v)


        out = {
            "o": o,
            "k": k,
            "q": q,
            # "e": e,
        }

        state = {
            "s": s_new,
        }

        return state, out
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method, i.e., before initialization
        
        batch_dims = x.shape[:-1]
        s = np.zeros(shape=(*batch_dims, self.qk_features, self.v_features), dtype=np.float32)

        return {
            "s": s,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
    
class TransformerStack(nn.Module):
    qk_features: Sequence[int]
    v_features: Sequence[int]
    stop_grad: bool = False

    def setup(self):
        layers = []
        for qk_f, v_f in zip(self.qk_features, self.v_features):
            layers.append(
                TransformerLayer(qk_features=qk_f, v_features=v_f)
            )
        self.layers = layers
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method
        states = []
        batch_shape = x.shape[:-1]
        for qk_f, v_f in zip(self.qk_features, self.v_features):
            s = np.zeros(shape=(*batch_shape, qk_f, v_f), dtype=np.float32)
            states.append({
                "s": s,
            })

        return states
    
    def out_to_next(self, out):
        x = out["o"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        return x
    
    def __call__(self, states, x):

        new_states, outs = [], []

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x)
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x = self.out_to_next(out)

        return new_states, outs

    def scan(self, states, x_seq):

        outs = []

        # here we just call sequentially the scan of each layer
        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_seq)
            outs.append(out)

            # set input sequence for the next layer
            x_seq = self.out_to_next(out)

        return outs
    
class TransformerStackClassifier(TransformerStack):

    out_labels: int = 1

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    
    def gather_input_classifier(self, outs):
        return outs[-1]["o"]
    
    def __call__(self, states, x):
        new_states, outs = super().__call__(states, x)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):
        outs = super().scan(states, x_seq)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return outs, aux
    

class CloneTestLayer(nn.Module):
    """
    Layer with HTM-like conceptual approach, to test different mathematical formulations
    """
    features : int
    clones : int

    def setup(self):
        self.forward = nn.Dense(self.features)
        self.lateral = nn.Dense(self.features*self.clones)

    def __call__(self, state, x):
        # Single time step, x assumed of shape (*, features)

        z_prev = state["z"]

        # Compute activation
        a_f = self.forward(x)
        a_l = self.lateral(z_prev)
        z_f = ut.threshold_softpp(
            a_f,
            a_min = 1.0/self.features,
            a_max = 1.0 - 1.0/self.features
        )
        z_l = ut.threshold_softpp(
            a_l,
            a_min = 1.0/(self.features),
            a_max = 1.0 - 1.0/(self.features)
        )
        z = z_f[..., None] * z_l.reshape((*z_l.shape[:-1], self.features, self.clones))
        # pool on clones
        y = z.max(-1)
        # flatten features and clones
        z = z.reshape((*z.shape[:-2], -1))

        out = {
            "z_f": z_f,
            "z_l": z_l,
            "z": z,
            "y": y,
        }
        
        state = {
            "z": z,
        }

        return state, out
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method, i.e., before initialization
        
        batch_dims = x.shape[:-1]
        mu = -2.0
        z = mu + jax.random.normal(key, shape=(*batch_dims, self.features*self.clones)).astype(np.float32)

        z = jax.nn.sigmoid(z)

        return {
            "z": z,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
    

class CloneTest(nn.Module):
    """
    A stack of CloneTestLayers
    """
    in_features: Sequence[int]
    features: Sequence[int]
    clones: Sequence[int]
    stop_grad: bool = False

    def setup(self):
        self.in_layers = [
            nn.Dense(in_f) for in_f in self.in_features
        ]
        self.layers = [
            CloneTestLayer(features=f, clones=c)
            for f, c in zip(self.features, self.clones)
        ]

    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state
        # Note: this method can be called without the "apply" method
        states = []
        batch_shape = x.shape[:-1]
        for f, c in zip(self.features, self.clones):
            key, _ = jax.random.split(key, 2)

            mu = -2.0            
            z = mu + jax.random.normal(key, shape=(*batch_shape, f*c)).astype(np.float32)
            z = jax.nn.sigmoid(z)
            
            states.append({
                "z": z,
            })

        return states

    def out_to_next(self, out):
        x = out["y"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        return x

    def __call__(self, states, x):

        new_states, outs = [], []

        # apply first layers
        x_in = x
        for i in range(len(self.in_layers)):
            x_in = self.in_layers[i](x_in)
            x_in = jax.nn.swish(x_in)

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x_in)
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x_in = self.out_to_next(out)

        return new_states, outs
    
    def scan(self, states, x_seq):

        outs = []

        ## apply first layers
        x_in = x_seq
        for i in range(len(self.in_layers)):
            x_in = self.in_layers[i](x_in)
            x_in = jax.nn.swish(x_in)

        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_in)
            outs.append(out)

            # set input sequence for the next layer
            x_in = self.out_to_next(out)

        return outs
    
class CloneTestClassifier(CloneTest):

    out_labels: int = 1

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    
    def gather_input_classifier(self, outs):
        x = outs[-1]["y"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        return x
    
    def __call__(self, states, x):
        new_states, outs = super().__call__(states, x)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):
        outs = super().scan(states, x_seq)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return outs, aux

class LinearStableLayer(nn.Module):
    features: int

    def setup(self,):
        # params to generate the recurrent linear matrix
        self.W = self.param(
            "W", 
            nn.initializers.normal(), 
            (self.features, self.features), 
            np.float32
        )
        # parameters for negative symmetric decay matrix
        self.W_dec = self.param(
            "W_dec", 
            nn.initializers.normal(), 
            (self.features, self.features), 
            np.float32
        )
        # simply a dense layer for the inputs
        self.B = nn.Dense(self.features)
        # output activation bias
        self.b = self.param(
            "b", 
            nn.initializers.constant(0.0), 
            (self.features,), 
            np.float32
        )

    def gen_recurrent_matrix(self, W, W_dec):
        # first create a skew-symmetric matrix (rotation component)
        A = 0.1*(W - W.T)
        # Generate decay component
        D = - W_dec.T @ W_dec
        # identity matrix
        I = np.eye(A.shape[-1])
        
        M = A + D

        term = I
        expr = I
        fact = 1.0
        order = 3
        
        for i in range(1, order + 1):
            fact *= i
            term = np.matmul(term, M)
            expr = expr + (term / fact)

        return expr
    
    def __call__(self, state, x):

        # s of shape (*batch_dims, qk_features, v_features)
        s = state["s"]

        # compute matrix for linear dynamical system
        M = self.gen_recurrent_matrix(self.W, self.W_dec)
        # compute recurrent state
        s_new = s @ M + self.B(x) 
        # compute non-linear activation
        z = jax.nn.sigmoid(s_new + self.b)

        new_state = {
            "s": s_new,
        }

        out = {
            "s": s_new,
            "z": z,
        }

        return new_state, out
    
    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state a
        # Note: this method can be called without the "apply" method, i.e., before initialization
        
        batch_dims = x.shape[:-1]
        s = np.zeros((*batch_dims, self.features)).astype(np.float32)

        return {
            "s": s,
        }
    
    def scan(self, state, x_seq):
        # assume input x_seq is of shape 
        # (*batch_dims, time, features), i.e., we have a sequence of time-steps

        def f_scan(crr, inp):
            x = inp
            state = crr
            new_state, outs = self(state, x)
            return new_state, outs

        # move time axis in firt position, required by jax.lax.scan
        x_seq = np.moveaxis(x_seq, -2, 0)

        # scan
        _, outs = jax.lax.scan(f_scan, state, x_seq)

        # move back time axis
        outs = jax.tree.map(lambda x: np.moveaxis(x, 0, -2), outs)

        return outs
    
class LinearStableStack(nn.Module):
    in_features: int
    features: Sequence[int]
    stop_grad: bool = False

    def setup(self):
        self.in_layer = nn.Dense(self.in_features)
        self.layers = [
            LinearStableLayer(features=f)
            for f in self.features
        ]

    @nn.nowrap
    def init_state_from_input(self, key, x):
        # Init a random recurrent state
        # Note: this method can be called without the "apply" method
        states = []
        batch_shape = x.shape[:-1]
        for f in self.features:
            key, _ = jax.random.split(key, 2)
            s = np.zeros((*batch_shape, f), dtype=np.float32)
            
            states.append({
                "s": s,
            })

        return states

    def out_to_next(self, out):
        x = out["z"]
        if self.stop_grad:
            x = jax.lax.stop_gradient(x)
        return x

    def __call__(self, states, x):

        new_states, outs = [], []

        # apply first layer
        x_in = self.in_layer(x)
        x_in = jax.nn.swish(x_in)

        for l_idx, layer in enumerate(self.layers):
            st, out = layer(states[l_idx], x_in)
            new_states.append(st)
            outs.append(out)

            # set input to next layer
            x_in = self.out_to_next(out)

        return new_states, outs
    
    def scan(self, states, x_seq):

        outs = []

        ## apply first layer
        x_in = self.in_layer(x_seq)
        x_in = jax.nn.swish(x_in)

        for l_idx, l in enumerate(self.layers):
            # note, scan does not return the final state
            out = l.scan(states[l_idx], x_in)
            outs.append(out)

            # set input sequence for the next layer
            x_in = self.out_to_next(out)

        return outs

class LinearStableClassifier(LinearStableStack):

    out_labels: int = 1
    stop_grad_classifier: bool = False

    def setup(self):
        super().setup()

        # final classifier layer
        self.classifier = nn.Dense(self.out_labels)

    
    def gather_input_classifier(self, outs):
        x = np.concatenate([o["z"] for o in outs], axis=-1)
        if self.stop_grad_classifier:
            x = jax.lax.stop_gradient(x)
        return x
    
    def __call__(self, states, x):
        new_states, outs = super().__call__(states, x)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return new_states, (outs, aux)
    
    def scan(self, states, x_seq):
        outs = super().scan(states, x_seq)
        logits = self.classifier(self.gather_input_classifier(outs))
        aux = {
            "logit": logits,
        }
        return outs, aux