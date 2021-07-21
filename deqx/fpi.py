import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from deqx import ad


class _FPIState(tp.NamedTuple):
    x: tp.Any
    step: tp.Any
    step_size: tp.Union[float, jnp.ndarray]
    iteration: tp.Union[int, jnp.ndarray]


class FPIInfo(tp.NamedTuple):
    step: tp.Any
    residual: float
    iterations: int


def fpi(
    fun: tp.Callable, x0: tp.Any, *args, tol: float = 1e-5, maxiter: int = 1000,
) -> tp.Tuple[tp.Any, FPIInfo]:
    """
    Fixed point iteration solver.

    Solves `x = fun(x)` starting from x0.

    Terminates at `maxiter` steps or when the 2-norm is less than `tol`.

    Args:
        fun: callable that takes x0 as an argument.
        x0: initial guess.
        tol: tolerance.
        maxiter: maximum number of iterations.

    Returns:
        sol: `x` s.t. `x = fun(x, *args)`
        FPIInfo: metadata from the final solver state.
    """
    tol2 = tol ** 2

    def cond(state: _FPIState):
        return jnp.logical_and(state.step_size > tol2, state.iteration < maxiter)

    def body(state: _FPIState):
        x = fun(state.x, *args)
        step = x - state.x
        step_size = sum(jnp.sum(jnp.square(x)) for x in jax.tree_leaves(step))
        return _FPIState(x, step, step_size, state.iteration + 1)

    state = _FPIState(x0, jax.tree_map(jnp.zeros_like, x0), jnp.inf, 0)
    state = jax.lax.while_loop(cond, body, state)
    return state.x, FPIInfo(state.step, state.step_size, state.iteration)


def fpi_with_jvp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get a fixed point solver with vjp implemented.

    Args:
        fun: function to find fixed point of
        jacobian_solver: linear system solver used in vjp.
        **kwargs: passed to `fpi`.

    Returns:
        function `z = ret_fun(x0, *args)` such that `fun(z, *args) == z` with
        implemented vjp.

    See also: `ad.fpi_with_vjp`
    """
    return ad.fpi_with_jvp(fun, partial(fpi, **kwargs), jacobian_solver)


def fpi_with_vjp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get a fixed point solver with jvp implemented.

    Args:
        fun: function to find fixed point of
        jacobian_solver: linear system solver used in jvp.
        **kwargs: passed to `newton`.

    Returns:
        function `z, info = ret_fun(x0, *args)` such that `fun(z, *args) == z` with
        implemented vjp.

    See also: `ad.fpi_with_jvp`
    """
    return ad.fpi_with_vjp(fun, partial(fpi, **kwargs), jacobian_solver=jacobian_solver)
