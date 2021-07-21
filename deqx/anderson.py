import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from deqx import ad


class _AndersonState(tp.NamedTuple):
    X: jnp.ndarray
    F: jnp.ndarray
    H: jnp.ndarray
    res: tp.Union[jnp.ndarray, float]
    k: tp.Union[jnp.ndarray, int]


class AndersonInfo(tp.NamedTuple):
    residual: jnp.ndarray
    iterations: int


def anderson(
    fun: tp.Callable,
    x0: tp.Any,
    *args,
    m: int = 5,
    lam: float = 1e-4,
    maxiter: int = 50,
    tol: float = 1e-5,
    beta: float = 1.0,
) -> tp.Tuple[tp.Any, AndersonInfo]:
    """
    Anderson accelerated fixed point iteration solver.

    Solves `x = fun(x)` starting from `x0`, based on pytorch implementation at
    https://implicit-layers-tutorial.org/deep_equilibrium_models/

    Terminates at `maxiter` steps or when the 2-norm is less than `tol`.

    Args:
        fun: callable that takes x0 as an argument.
        x0: initial guess.
        tol: tolerance.
        maxiter: maximum number of iterations.

    Returns:
        solution: `x` s.t. `x = fun(x)`
        FPIInfo
    """
    shape = x0.shape
    dtype = x0.dtype
    x0 = x0.reshape((-1,))
    size = x0.size

    def fx(x):
        return fun(x.reshape(shape), *args).reshape(-1)

    X = jnp.zeros((m, size), dtype=dtype).at[0].set(x0)
    F = jnp.zeros((m, size), dtype=dtype).at[0].set(fx(x0))
    H = jnp.zeros((m + 1, m + 1), dtype=dtype)
    H = H.at[0, 1:].set(1).at[1:, 0].set(1)
    state = _AndersonState(X, F, H, jnp.inf, 2)

    def cond(state: _AndersonState):
        return jnp.logical_and(state.k < maxiter, state.res > tol)

    def body(state: _AndersonState):
        X, F, H, _, k = state
        del state
        G = F - X
        H = H.at[1:, 1:].set(G @ G.T + lam * jnp.eye(m))
        y = jnp.zeros((m + 1,), dtype).at[0].set(1)
        alpha = jnp.linalg.solve(H, y)[1:]
        x = beta * (alpha @ F) + (1 - beta) * (alpha @ X)
        f = fx(x)
        # res = jnp.linalg.norm(f - x) / (1e-5 + jnp.linalg.norm(f))
        res = jnp.linalg.norm(f - x)
        X = X.at[k % m].set(x)
        F = F.at[k % m].set(f)

        return _AndersonState(X, F, H, res, k + 1)

    state = _AndersonState(X, F, H, jnp.inf, 0)
    state = jax.lax.while_loop(cond, body, state)
    return (
        state.X[(state.k - 1) % m].reshape(shape),
        AndersonInfo(state.res, state.k - 1),
    )


def anderson_with_vjp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get an `anderson` solver with vjp implemented.

    Args:
        fun: function to find fixed point of
        jacobian_solver: linear system solver used in vjp.
        **kwargs: passed to `anderson`.

    Returns:
        function `z = ret_fun(x0, *args)` such that `fun(z, *args) == z` with
        implemented vjp.

    See also: `ad.fpi_with_vjp`
    """
    return ad.fpi_with_vjp(fun, partial(anderson, **kwargs), jacobian_solver)


def anderson_with_jvp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get an `anderson` solver with jvp implemented.

    Args:
        fun: function to find fixed point of
        jacobian_solver: linear system solver used in jvp.
        **kwargs: passed to `anderson`.

    Returns:
        function `z, info = ret_fun(x0, *args)` such that `fun(z, *args) == z` with
        implemented vjp.

    See also: `ad.fpi_with_jvp`
    """

    return ad.fpi_with_jvp(fun, partial(anderson, **kwargs), jacobian_solver)
