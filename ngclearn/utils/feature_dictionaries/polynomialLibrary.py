from jax import jit, random, numpy as jnp
from typing import List, Tuple, Union
from dataclasses import dataclass

@dataclass
class PolynomialLibrary:
    """
    A class for creating polynomial feature libraries in 1D, 2D, or 3D.

    Args:
        poly_order (int): Maximum order of polynomial terms (Attribute)

        include_bias (bool): Whether to include the bias term in the output (Attribute)
    """

    poly_order: int = None
    include_bias: bool = True

    def __post_init__(self):
        if self.poly_order is None:
            raise ValueError("poly_order must be specified")
        if self.poly_order < 0 or not isinstance(self.poly_order, int):
            raise ValueError("poly_order must be an integer")


    def _create_library(self, *arrays: jnp.ndarray) -> Tuple[jnp.ndarray, List[str]]:
        """
        Create polynomial library for given input arrays.

        Args:
            arrays: Input arrays (x, y, z)

        Returns:
            Tuple of (feature matrix, feature names)
        """
        dim = len(arrays)
        lib = jnp.ones_like(arrays[0])
        names = ['1']

        if dim == 1:
            for i in range(self.poly_order + 1):
                lib = jnp.concatenate([lib, arrays[0] ** i], axis=1)
                if not (i == 0):
                    names.append(r'𝑥^{} |'.format(i))

        if dim == 2:
            for i in range(self.poly_order + 1):
                for j in range(self.poly_order - i + 1):
                    lib = jnp.concatenate([lib, arrays[0] ** i * arrays[1] ** j], axis=1)
                    if not (i == 0 and j == 0):
                        names.append(r'$𝑥^{} . 𝑦^{}$ |'.format(i, j))

        if dim == 3:
            for i in range(self.poly_order + 1):
                for j in range(self.poly_order + 1 - i):
                    for k in range(self.poly_order + 1 - (i + j)):
                        lib = jnp.concatenate([lib, arrays[0] ** i * arrays[1] ** j * arrays[2] ** k], axis=1)
                        if not (i == 0 and j == 0 and k == 0):
                            names.append(r'$𝑥^{} . 𝑦^{} . 𝓏^{}$ |'.format(i, j, k))

        return lib, names


    def fit(self, X: List[jnp.ndarray]) -> Tuple[jnp.ndarray, List[str]]:
        """
        Fits this library to a design matrix X

        Args:
            X: the design matrix to fit this library to

        Returns:
            the data-fit/retro-fit library
        """

        if not 1 <= len(X) <=3:
            raise ValueError("Input must be 1D, 2D, or 3D; e.g. len(X) >= 1 ")

        arrays = [jnp.array(x).reshape(-1, 1) for x in X]
        lib, names = self._create_library(*arrays)

        start_idx = 1 if not self.include_bias else 0
        return lib[:, start_idx+1:], names[start_idx:]

