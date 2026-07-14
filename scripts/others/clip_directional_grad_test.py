import jax
import jax.numpy as np
import infomax_sbdr as sbdr

x = np.array([-0.5, 0.5, 1.5])
# g simulates dL/d(out). Positive g at a saturated-low unit should be killed.
print(jax.vjp(sbdr.directional_clip, x)[1](np.array([1.0, 1.0, 1.0]))[0])
# -> [0., 1., 1.] : lower-saturated with g>0 killed, upper-saturated with g>0 kept
print(jax.vjp(sbdr.directional_clip, x)[1](np.array([-1.0, -1.0, -1.0]))[0])
# -> [-1., -1., 0.] : lower keeps g<0, upper-saturated with g<0 killed