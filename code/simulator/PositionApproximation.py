from typing import Final

from jax import config
from typing_extensions import Callable

config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  # pylint: disable=wrong-import-position


# HACK for some reason jnp.float64 calls asarray and takes too long, this way it is called only once
ZERO_TIME: Final[jnp.ndarray] = jnp.float64(0)


def constructPositionFunction(position: jnp.ndarray, velocity: jnp.ndarray, acceleration: jnp.ndarray)\
        -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Construct a position function approximation using Taylor.
    """

    def f(t: jnp.ndarray) -> jnp.ndarray:
        """
        This function simply satisfies:
            * f(0) = position
            * f'(0) = velocity
            * f''(0) = acceleration
        """
        #
        return position + velocity * t + (1/2) * acceleration * t ** 2

    return f
