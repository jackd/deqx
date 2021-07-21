import jax.numpy as jnp
import jax.test_util as jtu
from absl.testing import absltest

from deqx.anderson import anderson


class AndersonTest(jtu.JaxTestCase):
    def test_single_arg(self):
        def sqrt_fn(x, n):
            return x ** 2 - n + x

        n = jnp.asarray([0.5, 0.6])
        x0 = jnp.asarray([0.7, 0.7])

        sol, info = anderson(sqrt_fn, x0, n)
        self.assertAllClose(sol, -jnp.sqrt(n), atol=1e-5)

    def test_multi_args(self):
        def fun(x, a, b, c):
            return a * x * x + (b + 1) * x + c

        a = jnp.ones((2,))
        b = jnp.zeros((2,))
        c = -jnp.asarray([0.2, 0.3])

        x0 = jnp.asarray([0.2, 0.3])

        sol, _ = anderson(fun, x0, a, b, c)
        self.assertAllClose(sol, -jnp.sqrt(-c), atol=1e-5)

    def test_dict(self):
        def fun(x, params):
            return params["a"] * x * x + (1 + params["b"]) * x + params["c"]

        a = jnp.ones((2,))
        b = jnp.zeros((2,))
        c = -jnp.asarray([0.2, 0.3])
        params = dict(a=a, b=b, c=c)

        x0 = jnp.asarray([0.2, 0.3])

        sol, _ = anderson(fun, x0, params)
        self.assertAllClose(sol, -jnp.sqrt(-c), atol=1e-5)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
