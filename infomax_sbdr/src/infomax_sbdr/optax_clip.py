import optax
import jax
import jax.numpy as np

def optax_gen_clip_transform(min_val=None, max_val=None) -> optax.GradientTransformation:
    """An optax transform to keep parameters bounded within a given range."""
    
    def init_fn(params):
        # This transform doesn't require any state (like momentum or velocity)
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                "The custom `optax_clip` transform requires the current parameters to "
                "compute the bounded updates. Make sure to pass `params` to "
                "`optimizer.update()`."
            )

        # Function to apply to each leaf in the pytree
        def _clip_update(p, u):
            # Calculate what the parameter would be: p + u
            # Clip it to the desired bounds
            clipped_param = np.clip(p + u, a_min=min_val, a_max=max_val)
            # Return the adjusted update
            return clipped_param - p

        # Apply the clipping logic across the entire pytree of updates/params
        bounded_updates = jax.tree_util.tree_map(_clip_update, params, updates)
        
        return bounded_updates, state

    return optax.GradientTransformation(init_fn, update_fn)