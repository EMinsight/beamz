"""Static, exact support compression for single-device tensor updates."""

from dataclasses import dataclass

import jax
import numpy as np


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TensorRegion:
    """Shared-node coefficients on their fixed active bounding box.

    Bounds are static compilation metadata; values remain dynamic JAX leaves.
    The kernel adds an immutable zero halo for exact cropped averaging.
    """

    values: jax.Array | np.ndarray
    bounds: tuple[tuple[int, int], ...]
    node_shape: tuple[int, ...]

    @property
    def size(self):
        return self.values.size

    def tree_flatten(self):
        return (self.values,), (self.bounds, self.node_shape)

    @classmethod
    def tree_unflatten(cls, metadata, leaves):
        return cls(leaves[0], *metadata)


def compact_tensor_region(values):
    """Compress only when the halo-inclusive bounding box is substantially smaller.

    Exact zeros determine support: small physical cross coefficients are retained.
    Dense media retain the existing array representation and update path.
    """
    values = np.asarray(values)
    active = np.any(values != 0, axis=-1)
    if not np.any(active):
        return values
    bounds = []
    for axis in range(active.ndim):
        occupied = np.flatnonzero(
            np.any(active, axis=tuple(i for i in range(active.ndim) if i != axis))
        )
        bounds.append((int(occupied[0]), int(occupied[-1]) + 1))
    volume = np.prod(
        [
            min(count, stop + 1) - max(0, start - 1)
            for (start, stop), count in zip(bounds, active.shape, strict=True)
        ]
    )
    if volume > 0.5 * active.size:
        return values
    region = tuple(slice(start, stop) for start, stop in bounds)
    return TensorRegion(values[region].copy(), tuple(bounds), active.shape)
