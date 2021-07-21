import abc
import typing as tp

import jax.numpy as jnp
import jax.test_util as jtu
from absl.testing import absltest, parameterized

from deqx.anderson import anderson_with_jvp, anderson_with_vjp
from deqx.broyden import broyden_with_jvp, broyden_with_vjp
from deqx.fpi import fpi_with_jvp, fpi_with_vjp
from deqx.newton import newton_with_jvp, newton_with_vjp
from deqx.utils import to_fpi_fun


def sqrt_implicit(x, n):
    return x ** 2 - n


def quad_implicit(x, a, b, c):
    return a * x ** 2 + b * x + c


def quad_pos_analytic(a, b, c):
    return (-b + jnp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def quad_neg_analytic(a, b, c):
    return (-b - jnp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)


def neg_sqrt(x):
    return -jnp.sqrt(x)


sqrt_implicit_fpi = to_fpi_fun(sqrt_implicit)
quad_implicit_fpi = to_fpi_fun(quad_implicit)


sqrt_x0 = jnp.asarray([0.6, 0.7])
n = jnp.asarray([0.5, 0.6])
sqrt_args = (n,)

a = -jnp.asarray([1.0, 1.1])
b = -jnp.asarray([4.0, 4.1])
c = -jnp.asarray([0.5, 0.6])
quad_x0 = jnp.asarray([-1.0, -1.0])
quad_args = (a, b, c)


class ADTest(jtu.JaxTestCase):
    @abc.abstractmethod
    def solver_with_jvp(self, fun: tp.Callable) -> tp.Callable:
        raise NotImplementedError("Abstract method")

    @parameterized.parameters(
        (newton_with_jvp(sqrt_implicit), jnp.sqrt, sqrt_x0, sqrt_args),
        (broyden_with_jvp(sqrt_implicit), jnp.sqrt, sqrt_x0, sqrt_args),
        (fpi_with_jvp(sqrt_implicit_fpi), neg_sqrt, sqrt_x0, sqrt_args),
        (anderson_with_jvp(sqrt_implicit_fpi), neg_sqrt, sqrt_x0, sqrt_args),
        (newton_with_jvp(quad_implicit), quad_neg_analytic, quad_x0, quad_args),
        (broyden_with_jvp(quad_implicit), quad_neg_analytic, quad_x0, quad_args),
        # (fpi_with_jvp(quad_implicit_fpi), quad_pos_analytic, quad_x0, quad_args),
        (anderson_with_jvp(quad_implicit_fpi), quad_neg_analytic, quad_x0, quad_args),
    )
    def test_rootfinder_with_jvp(self, solver, analytic_fun, x0, args):
        expected = analytic_fun(*args)
        actual = solver(x0, *args)
        self.assertAllClose(actual, expected, rtol=1e-2)
        jtu.check_grads(
            lambda *args: solver(x0, *args)[0], args, order=1, modes=["fwd"], rtol=1e-2
        )

    @parameterized.parameters(
        (newton_with_vjp(sqrt_implicit), jnp.sqrt, sqrt_x0, sqrt_args),
        (broyden_with_vjp(sqrt_implicit), jnp.sqrt, sqrt_x0, sqrt_args),
        (fpi_with_vjp(sqrt_implicit_fpi), neg_sqrt, sqrt_x0, sqrt_args),
        (anderson_with_vjp(sqrt_implicit_fpi), neg_sqrt, sqrt_x0, sqrt_args),
        (newton_with_vjp(quad_implicit), quad_neg_analytic, quad_x0, quad_args),
        (broyden_with_vjp(quad_implicit), quad_neg_analytic, quad_x0, quad_args),
        # (fpi_with_vjp(quad_implicit_fpi), quad_pos_analytic, quad_x0, quad_args),
        (anderson_with_vjp(quad_implicit_fpi), quad_neg_analytic, quad_x0, quad_args),
    )
    def test_rootfinder_with_vjp(self, solver, analytic_fun, x0, args):
        expected = analytic_fun(*args)
        actual, info = solver(x0, *args)
        del info
        self.assertAllClose(actual, expected, rtol=1e-2)
        jtu.check_grads(
            lambda *args: solver(x0, *args)[0], args, order=1, modes=["rev"], rtol=1e-2
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
