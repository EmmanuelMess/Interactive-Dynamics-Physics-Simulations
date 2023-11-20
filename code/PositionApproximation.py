from jax import config, jit

config.update("jax_enable_x64", True)

import jax.numpy as jnp


def constructPositionFunction(position: jnp.ndarray, velocity: jnp.ndarray, acceleration: jnp.ndarray):
    """
    Construct a position function approximation using Taylor.
    """

    @jit
    def f(t: jnp.ndarray):
        """
        This function simply satisfies:
            * f(0) = position
            * f'(0) = velocity
            * f''(0) = acceleration
        """
        return position + velocity * t + (1/2) * acceleration * t ** 2

    return f
