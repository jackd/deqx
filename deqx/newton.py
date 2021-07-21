import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from deqx import ad


class NewtonState(tp.NamedTuple):
    """State of newton solver."""

    x: jnp.ndarray  # solution, [n]float
    residual: jnp.ndarray  # function evaluated at `x`, [n]float
    err: jnp.ndarray  # sum-of-squares of residual []float
    iteration: jnp.ndarray  # []int


class NewtonInfo(tp.NamedTuple):
    residual: jnp.ndarray
    err: jnp.ndarray
    iterations: jnp.ndarray


def newton(
    g: tp.Callable,
    x0: jnp.ndarray,
    *args,
    maxiter: int = 10000,
    tol: float = 1e-5,
    atol: float = 0.0,
    solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> tp.Tuple[jnp.ndarray, NewtonInfo]:
    """
    Find `x` such that `g(x, *args) == 0`.

    Args:
        x0: initial guess of solution.
        *args: other args to use in `g`.
        maxiter: maximum number of iterations.
        solver: Callable like gmres.

    Returns:
        (sol, info) such that `fun(sol, *args) == 0`.
    """

    def gx(x):
        return g(x, *args)

    def err_fun(residual):
        return sum(jnp.sum(jnp.square(x)) for x in jax.tree_leaves(residual))

    residual = gx(x0)
    err = err_fun(residual)
    atol2 = jnp.maximum(jnp.square(tol) * err, jnp.square(atol))

    def cond(state: NewtonState):
        return (state.err > atol2) & (state.iteration < maxiter)

    def body(state: NewtonState):
        def Jv(v):
            primals_out, tangents_out = jax.jvp(gx, (state.x,), (v,))
            del primals_out
            return tangents_out

        step, _ = solver(Jv, state.residual)
        x = state.x - step
        residual = gx(x)
        err = err_fun(residual)
        return NewtonState(x, residual, err, state.iteration + 1)

    state = NewtonState(x0, residual, err, jnp.zeros((), jnp.int32))
    state = jax.lax.while_loop(cond, body, state)
    return state.x, NewtonInfo(state.residual, state.err, state.iteration)


def newton_with_vjp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get a newton solver with vjp implemented.

    Args:
        fun: function to find roots of
        jacobian_solver: linear system solver used in vjp.
        **kwargs: passed to `newton`.

    Returns:
        function `z, info = ret_fun(x0, *args)` such that `fun(z, *args) == 0` with
        implemented vjp.

    See also: `ad.rootfinder_with_vjp`
    """
    return ad.rootfinder_with_vjp(fun, partial(newton, **kwargs), jacobian_solver)


def newton_with_jvp(
    fun: tp.Callable,
    *,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
    **kwargs,
) -> tp.Callable:
    """
    Get a newton solver with jvp implemented.

    Args:
        fun: function to find roots of
        jacobian_solver: linear system solver used in vjp.
        **kwargs: passed to `newton`.

    Returns:
        function `z = ret_fun(x0, *args)` such that `fun(z, *args) == 0` with
        implemented jvp.

    See also: `ad.rootfinder_with_jvp`
    """
    return ad.roofinder_with_jvp(fun, partial(newton, **kwargs), jacobian_solver)
