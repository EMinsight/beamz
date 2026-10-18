"""JAX test helpers compatible with the supported Python/JAX versions."""

try:
    from jax import enable_x64 as enable_x64
except ImportError:
    from jax.experimental import enable_x64 as enable_x64
