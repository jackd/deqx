# Deep Equilibrium Models in Jax

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[jax][jax] implementations of rootfind and fixed-point solvers, along with vector-jacobian products and jacobian-vector products and Deep Equilibrium (DEQ) layers.

- rootfind algorithms:
  - [newton](./deqx/newton.py)
  - [broyden](./deqx/broyden.py) (based on [this work][original])
- fixed point algorithms:
  - [fixed point iteration](./deqx/fpi.py)
  - [anderson](./deqx/anderson.py)
- [automatic differentiation](./deqx/ad.py) - `jvp`s and `vjp`s
- [Deep Equilibrium Layers](./deqx/deq.py) ([haiku](https://github.com/deepmind/dm-haiku))

## Installation

```bash
pip install jax  # cpu only - see https://github.com/google/jax for GPU installation
pip install dm-haiku
git clone https://github.com/jackd/deqx.git
pip install -e deqx # local install
```

## Example Usage

See the [test](./test) directory for low-level usage. For a full network example using haiku see [mnist.py](./examples/mnist.py) (disclaimer: it runs slowly and results in poor accuracy. Issues/PRs that improve upon this will be greatly appreciated).

```bash
pip install tensorflow tensorflow-datasets # used for data
python deqx/examples/mnist.py
```

The below is an excerpt for building the model.

```python
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp

from deqx.deq import DEQ
from deqx.newton import newton_with_vjp


def fpi_fun(z, x):
    conv = hk.Conv2D(num_features, 3, 1, w_init=hk.initializers.TruncatedNormal(1e-2))
    z = jax.nn.relu(conv(z) + x)
    z = hk.LayerNorm((1, 2), True, True)(z)
    return x


def model_fn(x):
    x = hk.Conv2D(num_features, 5, 2)(x)
    x = jax.nn.relu(x)
    x = hk.LayerNorm((1, 2), True, True)(x)
    x = DEQ(
        fpi_fun,
        partial(
            newton_with_vjp,
            tol=1e-3,
            jacobian_solver=partial(jax.scipy.sparse.linalg.gmres, tol=1e-3),
        ),
    )(jnp.zeros_like(x), x)

    x = jnp.mean(x, axis=(1, 2))  # spatial pooling
    x = hk.Linear(10)(x)
    return x
```

## Tests

```bash
pip install pytest
pytest deqx/test/
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```

[original-jax]:https://github.com/akbir/deq-jax
[jax]:https://github.com/google/jax
