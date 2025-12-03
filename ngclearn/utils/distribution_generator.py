import time
from typing import TypedDict, List, Protocol, Sequence
from typing_extensions import Unpack
import jax
import numpy

from ngcsimlib.logger import error


class DistributionParams(TypedDict, total=False):
    """
    Extra parameters to be used when generating distributions. (Attributes listed below)

    Args:
        amin: sets the lower bound of the distribution

        amax: sets the upper bound of the distribution

        lower_triangle: keeps the lower triangle, sets the rest to zero

        upper_triangle: keeps the upper triangle, sets the rest to zero

        hollow: produces a hollow distribution (zeros along the diagonal)

        eye: produces an eye distribution (zeros the off-diagonal)

        col_mask: single value, keeps n random columns; list values, keeps the provided column indices

        row_mask: single value, keeps n random rows; list values, keeps the provided row indices

        use_numpy: use default numpy

    """
    amin: float
    amax: float
    lower_triangle: bool
    upper_triangle: bool
    hollow: bool
    eye: bool
    col_mask: int | List[int]
    row_mask: int | List[int]
    use_numpy: bool
    dtype: numpy.dtype


class DistributionInitializer(Protocol):
    def __call__(self, shape: Sequence[int], dkey: jax.dtypes.prng_key | int | None = None) -> jax.Array: ...


class DistributionGenerator(object):
    @staticmethod
    def constant(value: float, **params: Unpack[DistributionParams]) -> DistributionInitializer:
        """
        Produces a distribution initializer for a constant distribution.

        Args:
            value: the constant value to fill the array with
            **params: the extra distribution parameters

        Returns:
            a distribution initializer
        """
        using_np = params.get("use_numpy", False)
        if using_np:
            def constant_generator(shape: Sequence[int], seed: int | None = None) -> numpy.ndarray:
                matrix = numpy.ones(shape, params.get("dtype", numpy.float32)) * value
                matrix = DistributionGenerator._process_params_numpy(matrix, params, seed)
                return matrix
        else:
            def constant_generator(shape: Sequence[int], dKey: jax.dtypes.prng_key | None = None) -> jax.Array:
                matrix = jax.numpy.ones(shape, params.get("dtype", jax.numpy.float32)) * value
                matrix = DistributionGenerator._process_params_jax(matrix, params, dKey)
                return matrix
        return constant_generator

    @staticmethod
    def uniform(low: float = 0.0, high: float = 1.0, **params: Unpack[DistributionParams]) -> DistributionInitializer:
        """
        Produces a distribution initializer for a uniform distribution.

        Args:
            low: lower bound of the uniform distribution (inclusive)
            high: upper bound of the uniform distribution (exclusive)
            **params: the extra distribution parameters

        Returns:
            a distribution initializer
        """
        using_np = params.get("use_numpy", False)

        if using_np:
            def uniform_generator(shape: Sequence[int], seed: int | None = None) -> numpy.ndarray:
                rng = numpy.random.default_rng(seed)
                matrix = rng.uniform(low=low, high=high, size=shape).astype(
                    params.get("dtype", numpy.float32))
                matrix = DistributionGenerator._process_params_numpy(matrix, params, seed)
                return matrix
        else:
            def uniform_generator(shape: Sequence[int], dKey: jax.Array | None = None) -> jax.Array:
                if dKey is None:
                    dKey = jax.random.PRNGKey(time.time_ns())
                dKey, subKey = jax.random.split(dKey, 2)

                matrix = jax.random.uniform(
                    dKey,
                    shape=shape,
                    minval=low,
                    maxval=high,
                    dtype=params.get("dtype", jax.numpy.float32)
                )
                matrix = DistributionGenerator._process_params_jax(matrix, params, subKey)
                return matrix

        return uniform_generator

    @staticmethod
    def gaussian(mean: float = 0.0, std: float = 1.0, **params: Unpack[DistributionParams]) -> DistributionInitializer:
        """
        Produces a distribution initializer for a Gaussian (normal) distribution.

        Args:
            mean: the mean of the normal distribution
            std: the standard deviation of the normal distribution
            **params: the extra distribution parameters

        Returns:
            a distribution initializer
        """
        using_numpy = params.get("use_numpy", False)

        if using_numpy:
            def gaussian_generator(shape: Sequence[int], seed: int | None = None) -> numpy.ndarray:
                rng = numpy.random.default_rng(seed)
                matrix = rng.normal(loc=mean, scale=std, size=shape).astype(
                    params.get("dtype", numpy.float32))
                matrix = DistributionGenerator._process_params_numpy(matrix, params, seed)
                return matrix
        else:
            def gaussian_generator(shape: Sequence[int], dKey: jax.Array | None = None) -> jax.Array:
                if dKey is None:
                    dKey = jax.random.PRNGKey(time.time_ns())
                dKey, subKey = jax.random.split(dKey, 2)
                matrix = jax.random.normal(
                    dKey,
                    shape=shape,
                    dtype=params.get("dtype", jax.numpy.float32)
                )
                matrix = mean + std * matrix
                matrix = DistributionGenerator._process_params_jax(matrix, params, subKey)
                return matrix

        return gaussian_generator

    @staticmethod
    def fan_in_uniform(**params: Unpack[DistributionParams]) -> DistributionInitializer:
        """
        Produces a distribution initializer using a fan-in uniform strategy.
        The values are sampled from a uniform distribution in the range [-limit, limit],
        where limit = sqrt(1 / fan_in), and fan_in is inferred from the shape.

        | Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of training deep feedforward neural
        | networks." Proceedings of the thirteenth international conference on artificial intelligence and statistics.
        | JMLR Workshop and Conference Proceedings, 2010.

        Args:
            **params: extra distribution parameters

        Returns:
            a distribution initializer
        """
        using_numpy = params.get("use_numpy", False)

        def compute_limit(fan_in: int) -> float:
            return float(numpy.sqrt(1.0 / fan_in))

        if using_numpy:
            def fan_in_uniform_generator(shape: Sequence[int], seed: int | None = None) -> numpy.ndarray:
                if len(shape) < 2:
                    error("fan_in_uniform requires shape with at least 2 dimensions")
                fan_in = shape[1]
                limit = compute_limit(fan_in)

                rng = numpy.random.default_rng(seed)
                matrix = rng.uniform(low=-limit, high=limit, size=shape).astype(
                    params.get("dtype", numpy.float32))
                matrix = DistributionGenerator._process_params_numpy(matrix, params, seed)
                return matrix
        else:
            def fan_in_uniform_generator(shape: Sequence[int], dKey: jax.Array | None = None) -> jax.Array:
                if len(shape) < 2:
                    error("fan_in_uniform requires shape with at least 2 dimensions")
                fan_in = shape[1]
                limit = compute_limit(fan_in)

                if dKey is None:
                    dKey = jax.random.PRNGKey(time.time_ns())
                dKey, subKey = jax.random.split(dKey, 2)

                matrix = jax.random.uniform(
                    dKey,
                    shape=shape,
                    minval=-limit,
                    maxval=limit,
                    dtype=params.get("dtype", jax.numpy.float32)
                )
                matrix = DistributionGenerator._process_params_jax(matrix, params, subKey)
                return matrix

        return fan_in_uniform_generator

    @staticmethod
    def fan_in_gaussian(**params: Unpack[DistributionParams]) -> DistributionInitializer:
        """
        Produces a distribution initializer using a fan-in Gaussian (normal) strategy.
        The values are sampled from a normal distribution with mean 0 and stddev = sqrt(1 / fan_in),
        where fan_in is inferred from the shape.

        | He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level performance on imagenet
        | classification." Proceedings of the IEEE international conference on computer vision. 2015.

        Args:
            **params: extra distribution parameters

        Returns:
            a distribution initializer
        """
        using_numpy = params.get("use_numpy", False)

        def compute_std(fan_in: int) -> float:
            return float(numpy.sqrt(1.0 / fan_in))

        if using_numpy:
            def fan_in_gaussian_generator(shape: Sequence[int], seed: int | None) -> numpy.ndarray:
                if len(shape) < 2:
                    error("fan_in_gaussian requires shape with at least 2 dimensions")
                fan_in = shape[0]
                std = compute_std(fan_in)

                rng = numpy.random.default_rng(seed)
                matrix = rng.normal(loc=0.0, scale=std, size=shape).astype(
                    params.get("dtype", numpy.float32))
                matrix = DistributionGenerator._process_params_numpy(matrix, params, seed)
                return matrix
        else:
            def fan_in_gaussian_generator(shape: Sequence[int], dKey: jax.Array | None) -> jax.Array:
                if len(shape) < 2:
                    error("fan_in_gaussian requires shape with at least 2 dimensions")
                fan_in = shape[0]
                std = compute_std(fan_in)

                if dKey is None:
                    dKey = jax.random.PRNGKey(time.time_ns())
                dKey, subKey = jax.random.split(dKey, 2)

                matrix = jax.random.normal(
                    dKey,
                    shape=shape,
                    dtype=params.get("dtype", jax.numpy.float32)
                )
                matrix = matrix * std
                matrix = DistributionGenerator._process_params_jax(matrix, params, subKey)
                return matrix

        return fan_in_gaussian_generator

    @staticmethod
    def _process_params_jax(ary: jax.Array, params: DistributionParams, dKey: jax.dtypes.prng_key | None) -> jax.Array:
        if dKey is None:
            dKey = jax.random.PRNGKey(time.time_ns())

        amin = params.get("amin", None)
        if amin is not None:
            ary = jax.numpy.maximum(ary, amin)

        amax = params.get("amax", None)
        if amax is not None:
            ary = jax.numpy.minimum(ary, amax)

        lower_triangle = params.get("lower_triangle", False)
        upper_triangle = params.get("upper_triangle", False)
        if lower_triangle and upper_triangle:
            error("lower_triangle and upper_triangle are mutually exclusive when initializing a distribution")

        if lower_triangle:
            ary = jax.numpy.tril(ary)
        if upper_triangle:
            ary = jax.numpy.triu(ary)

        if params.get("hollow", False):
            ary = (1.0 - jax.numpy.eye(*ary.shape)) * ary

        if params.get("eye", False):
            ary = jax.numpy.eye(*ary.shape) * ary

        col_mask = params.get("col_mask", None)
        if col_mask is not None:
            if isinstance(col_mask, int):
                dKey, subKey = jax.random.split(dKey, 2)
                keep_indices = jax.random.choice(subKey, ary.shape[1], shape=(col_mask,), replace=False)
                mask = jax.numpy.zeros(ary.shape[1], dtype=bool).at[
                    keep_indices].set(True)
                mask = jax.numpy.broadcast_to(mask, ary.shape)
                ary = jax.numpy.where(mask, ary, 0)
            elif isinstance(col_mask, Sequence):
                mask = jax.numpy.zeros(ary.shape[1], dtype=bool).at[
                    col_mask].set(True)
                mask = jax.numpy.broadcast_to(mask, ary.shape)
                ary = jax.numpy.where(mask, ary, 0)

        row_mask = params.get("row_mask", None)
        if row_mask is not None:
            if isinstance(row_mask, int):
                dKey, subKey = jax.random.split(dKey, 2)
                keep_indices = jax.random.choice(subKey, ary.shape[0], shape=(row_mask,), replace=False)
                mask = jax.numpy.zeros(ary.shape[0], dtype=bool).at[
                    keep_indices].set(True)
                mask = jax.numpy.broadcast_to(mask, ary.shape)
                ary = jax.numpy.where(mask, ary, 0)
            elif isinstance(row_mask, Sequence):
                mask = jax.numpy.zeros(ary.shape[0], dtype=bool).at[
                    row_mask].set(True)
                mask = jax.numpy.broadcast_to(mask, ary.shape)
                ary = jax.numpy.where(mask, ary, 0)

        return ary.astype(params.get("dtype", jax.numpy.float32))

    @staticmethod
    def _process_params_numpy(ary: numpy.ndarray, params: DistributionParams, seed: int | None) -> numpy.ndarray:
        amin = params.get("amin", None)
        if amin is not None:
            ary = numpy.maximum(ary, amin)

        amax = params.get("amax", None)
        if amax is not None:
            ary = numpy.minimum(ary, amax)

        lower_triangle = params.get("lower_triangle", False)
        upper_triangle = params.get("upper_triangle", False)
        if lower_triangle and upper_triangle:
            error("lower_triangle and upper_triangle are mutually exclusive when initializing a distribution")

        if lower_triangle:
            ary = numpy.tril(ary)
        if upper_triangle:
            ary = numpy.triu(ary)

        if params.get("hollow", False):
            ary = (1.0 - numpy.eye(*ary.shape)) * ary

        if params.get("eye", False):
            ary = numpy.eye(*ary.shape) * ary

        col_mask = params.get("col_mask", None)
        if col_mask is not None:
            if isinstance(col_mask, int):
                rng = numpy.random.default_rng(seed)
                keep_indices = rng.choice(ary.shape[1], size=col_mask, replace=False)
                mask = numpy.zeros(ary.shape[1], dtype=bool)
                mask[keep_indices] = True
                mask = numpy.broadcast_to(mask, ary.shape)
                ary = numpy.where(mask, ary, 0)
            elif isinstance(col_mask, Sequence):
                mask = numpy.zeros(ary.shape[1], dtype=bool)
                mask[list(col_mask)] = True
                mask = numpy.broadcast_to(mask, ary.shape)
                ary = numpy.where(mask, ary, 0)

        row_mask = params.get("row_mask", None)
        if row_mask is not None:
            if isinstance(row_mask, int):
                rng = numpy.random.default_rng(seed)
                keep_indices = rng.choice(ary.shape[0], size=row_mask, replace=False)
                mask = numpy.zeros(ary.shape[0], dtype=bool)
                mask[keep_indices] = True
                mask = numpy.broadcast_to(mask, ary.shape)
                ary = numpy.where(mask, ary, 0)
            elif isinstance(row_mask, Sequence):
                mask = numpy.zeros(ary.shape[0], dtype=bool)
                mask[list(row_mask)] = True
                mask = numpy.broadcast_to(mask, ary.shape)
                ary = numpy.where(mask, ary, 0)

        return ary
