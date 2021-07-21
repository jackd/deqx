import typing as tp
from functools import partial

import jax
import jax.numpy as jnp

from deqx import utils


def rootfind_vjp(
    fun: tp.Callable,
    root: jnp.ndarray,
    args: tp.Tuple,
    g: jnp.ndarray,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> tp.Tuple:
    """
    rootfind vjp implementation computed using vector inverse Jacobian vector product.

    Args:
        fun: Callable such that `jnp.all(fun(root, *args) == 0)`
        root: x such that fun(x, *args) == 0
        args: tuple, used in fun.
        g: gradient of some scalar (e.g. a neural network loss) w.r.t. root.
        solver: approximate linear system solver, `solver(A, b) == x` x s.t. A(x) = b.

    Returns:
        vjp of root w.r.t. args, same structure as args.
    """
    _, vjp_fun = jax.vjp(lambda x: fun(x, *args), root)
    vJ, _ = jacobian_solver(lambda x: vjp_fun(x)[0], -g)
    out = jax.vjp(partial(fun, root), *args)[1](vJ)
    return out


def rootfinder_with_vjp(
    fun: tp.Callable, rootfind_solver: tp.Callable, jacobian_solver: tp.Callable
) -> tp.Callable:
    """
    Create a rootfind function with vector-jacobian product (vjp) implemented.

    Args:
        fun: function to find roots of, `fun(x, *args)`
        rootfind_solver: rootfind algorithm, e.g. `deqx.newton.newton`
        jacobian_solver: linear system solver used to compute vjp.

    Returns:
        function that finds `x` such that `fun(x, *args) == 0`.
    """

    @jax.custom_vjp
    def rootfind(x, *args):
        return rootfind_solver(fun, x, *args)

    def fwd(x, *args):
        root, info = rootfind(x, *args)
        return (root, info), (root, args)

    def rev(res, g):
        root, args = res
        root_g, info_g = g
        del info_g
        arg_grads = rootfind_vjp(fun, root, args, root_g, jacobian_solver)
        return (None, *arg_grads)

    rootfind.defvjp(fwd, rev)
    return rootfind


def fpi_vjp(
    fun: tp.Callable,
    sol: jnp.ndarray,
    args: tp.Tuple,
    g: jnp.ndarray,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> tp.Tuple:
    """
    vector-jacobian product (vjp) of fpi function.

    Args:
        fun: function to find fixed point of, fun(x, *args)
        sol: solution, sol == fun(sol, *args)
        g: vector on left of vjp.
        jacobian_solver: linear solver.

    Returns:
        vector-jacobian product.

    See also: `rootfind_vjp`
    """
    return rootfind_vjp(utils.to_rootfind_fun(fun), sol, args, g, jacobian_solver)

    # _, vjp_fun = jax.vjp(lambda x: fun(x, *args), sol)
    # vJ, _ = solver(lambda x: x - vjp_fun(x)[0], g)
    # out = jax.vjp(partial(fun, sol), *args)[1](vJ)
    # return out


def fpi_with_vjp(
    fun: tp.Callable,
    fpi_solver: tp.Callable,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> tp.Callable:
    """
    Create a fixed point function with vector-jacobian product (vjp) implemented.

    Args:
        fun: function to find roots of, `fun(x, *args)`
        fpi_solver: fpi algorithm, e.g. `deqx.fpi.fpi`
        jacobian_solver: linear system solver used to compute vjp.

    Returns:
        `ret_fun` such that `sol = ret_fun(x0, *args)` such that
        `fun(sol, *args) == sol` with vjp implemented.
    """

    @jax.custom_vjp
    def fpi_fun(x0, *args):
        return fpi_solver(fun, x0, *args)

    def fwd(x0, *args):
        sol, info = fpi_fun(x0, *args)
        return (sol, info), (sol, args)

    def rev(res, g):
        sol, args = res
        root_g, info_g = g
        del info_g
        arg_grads = fpi_vjp(fun, sol, args, root_g, jacobian_solver)
        return (None, *arg_grads)

    fpi_fun.defvjp(fwd, rev)
    return fpi_fun


def rootfind_jvp(
    fun: tp.Callable,
    root: jnp.ndarray,
    args: tp.Tuple,
    tangents: tp.Tuple,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> jnp.ndarray:
    """
    Get jvp of root w.r.t. args.

    Args:
        fun: function with found root.
        root: root of fun such that `fun(root, *args) == 0`.
        args: additional arguments to `fun`.
        tangents: tangents corresponding to `args`.
        jacobian_solver: linear system solver, i.e. if `x = jacobian_solver(A, b)` then
            `A(x) == b`.

    Returns:
        jvp of `root` given `tangents` of `args`.
    """
    if len(args) == 0:
        # if there are no other parameters there will be no gradient
        return ()

    # fun_dot is the jvp of fun w.r.t all `*args`
    _, fun_dot = jax.jvp(partial(fun, root), args, tangents)

    def Jx(v):
        # The Jacobian of fun(x, *args) w.r.t. x evaluated at (primal_out, *args)
        return jax.jvp(lambda x: fun(x, *args), (root,), (v,))[1]

    sol, _ = jacobian_solver(Jx, fun_dot)
    tangent_out = -sol
    return tangent_out


def fpi_jvp(
    fun: tp.Callable,
    sol: jnp.ndarray,
    args: tp.Tuple,
    tangents: tp.Tuple,
    jacobian_solver: tp.Callable = jax.scipy.sparse.linalg.gmres,
) -> jnp.ndarray:
    """
    Get jvp of sol w.r.t. args.

    Args:
        fun: function with found fixed point.
        sol: fixed point solution, i.e. `fun(sol, *args) == sol`.
        args: additional arguments to `fun`.
        tangents: tangents corresponding to `args`.
        jacobian_solver: linear system solver, i.e. if `x = jacobian_solver(A, b)` then
            `A(x) == b`.

    Returns:
        jvp of `sol` given `tangents` of `args`.
    """
    return rootfind_jvp(
        utils.to_rootfind_fun(fun), sol, args, tangents, jacobian_solver
    )
    # if len(args) == 0:
    #     # output should be discontinuous/non-differentiable w.r.t initial solution
    #     # if there are no other parameters there will be no gradient
    #     return ()

    # # fun_dot is the jvp of fun w.r.t all `*args`
    # _, fun_dot = jax.jvp(partial(fun, sol), args, tangents)

    # def Jx(v):
    #     # The Jacobian of fun(x, *args) w.r.t. x evaluated at (primal_out, *args)
    #     return jax.jvp(lambda x: x - fun(x, *args), (sol,), (v,))[1]

    # sol, _ = solver(Jx, fun_dot)
    # tangent_out = sol
    # return tangent_out


def roofinder_with_jvp(
    fun: tp.Callable, rootfind_solver: tp.Callable, jacobian_solver: tp.Callable
):
    """
    Get a rootfinder function with jvp implemented.

    Args:
        fun: function to find roots of.
        rootfind_solver: rootfind algorithm, e.g. `deqx.newton.newton`
        jacobian_solver: linear solver.

    Returns:
        `solver` such that `z = solver(x0, *args)` satisfies `fun(z, *args) == 0`.
        `solver` has a `jvp` which is computed using `jacobian_solver`.
    """

    @jax.custom_jvp
    def rootfind(x, *args):
        # return rootfind_fun(fun, x, *args)
        sol, info = rootfind_solver(fun, x, *args)
        del info
        return sol

    @rootfind.defjvp
    def jvp(primals, tangents):
        """
        Calculate jacobian-vector product using implicit differentiation.

        Denoting `fun` as `g`, `x_star` as the root and grouping `*args` as `theta`:

        g(x_star(theta), theta) = 0 (eqn 1)

        then differentiating eqn 1 w.r.t theta and rearranging yields:

        dx_star/dtheta = -[dg/dx]^{-1} dg/dtheta
        hence
        tangent_out = inv(J_x(x_star, theta)) @ J_theta(x_star, theta) @ theta_dot
        """
        x, *args = primals
        xt, *tangents = tangents
        del xt  # solution is not differentiable w.r.t initial guess
        # primal_out, info = rootfind(x, *args)
        # tangent_out = rootfind_jvp(
        #     fun, primal_out, args, tangents, jacobian_solver=jacobian_solver
        # )
        # return (primal_out, info), (tangent_out, jax.tree_map(jnp.zeros_like, info))
        primal_out = rootfind(x, *args)
        tangent_out = rootfind_jvp(
            fun, primal_out, args, tangents, jacobian_solver=jacobian_solver
        )
        return primal_out, tangent_out

    return rootfind


def fpi_with_jvp(
    fun: tp.Callable, fpi_solver: tp.Callable, jacobian_solver: tp.Callable
) -> tp.Callable:
    """
    Create a fixed-point solver with jacobian-vector product (jvp).

    Args:
        fun: function to find fixed point of, i.e. `x` such that `fun(x, *args) == 0`.
        fpi_solver: fixed-point solver, e.g. `deqx.fpi.fpi`.
        jacobian_solver: linear system solver used to compute jvp.

    Returns:
        fixed point solver that uses `fpi_solver` to solve the fixed point and with `jvp`
        implementation using `jacobian_solver`.
    """

    @jax.custom_jvp
    def fpi_solve(x, *args):
        sol, info = fpi_solver(fun, x, *args)
        del info
        return sol

    @fpi_solve.defjvp
    def fpi_solve_jvp(primals, tangents):
        x, *args = primals
        xt, *tangents = tangents
        del xt
        # primal_out, info = fpi_solve(x, *args)
        primal_out = fpi_solve(x, *args)
        tangent_out = fpi_jvp(fun, primal_out, args, tangents, jacobian_solver)
        # tangent_info = jax.tree_map(lambda i: Zero(i.aval), info)
        return primal_out, tangent_out
        # return (primal_out, info), (tangent_out, tangent_info)

    return fpi_solve
