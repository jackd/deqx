import haiku as hk
import jax
import jax.numpy as jnp
import jax.test_util as jtu
from absl.testing import absltest

from deqx.deq import DEQ, SolverType
from deqx.newton import newton_with_vjp


def resnet_simple(z, x):
    linear = hk.Linear(x.shape[-1], w_init=hk.initializers.TruncatedNormal(1e-2))
    return jax.nn.relu(linear(z) + x)


class DEQTest(jtu.JaxTestCase):
    def test_deq(self):
        def deq_fun(x):
            return DEQ(resnet_simple, newton_with_vjp, solver_type=SolverType.ROOT)(
                jnp.zeros_like(x), x
            )

        transform = hk.transform(deq_fun)
        rng = hk.PRNGSequence(0)
        x = jax.random.normal(next(rng), (5, 7))

        params = transform.init(next(rng), x)
        jtu.check_grads(
            lambda x, params: transform.apply(params, next(rng), x),
            (x, params),
            order=1,
            modes=["rev"],
        )

        # ensure z satisfies fixed point condition
        z = transform.apply(params, next(rng), x)
        z2 = hk.transform(resnet_simple).apply(
            {"linear": params["deq/lifted/linear"]}, next(rng), z, x
        )
        self.assertAllClose(z, z2)

    def test_deq_model(self):
        def deq_fun(x):
            x = hk.Linear(10)(x)
            x = jax.nn.relu(x)
            return DEQ(resnet_simple, newton_with_vjp, solver_type=SolverType.ROOT)(
                jnp.zeros_like(x), x
            )

        transform = hk.transform(deq_fun)
        rng = hk.PRNGSequence(0)
        x = jax.random.normal(next(rng), (5, 7))

        params = transform.init(next(rng), x)
        jtu.check_grads(
            lambda x, params: transform.apply(params, next(rng), x),
            (x, params),
            order=1,
            modes=["rev"],
        )


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
