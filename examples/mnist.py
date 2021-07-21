"""
mnist example using haiku and a DEQ layer.

This model trains very poorly - whether that is down to a bug in the implementation or
just a poorly designed model is an open question. Github issues that identify the
problem (or better yet, PRs that resolve it) will be welcomed.
"""
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app

from deqx.deq import DEQ
from deqx.newton import newton_with_vjp

batch_size = 64
num_features = 256


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


def sparse_categorical_crossentropy_from_logits(
    labels: jnp.ndarray, logits: jnp.ndarray
) -> jnp.ndarray:
    preds = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(preds, labels[..., None], axis=-1)[..., 0]
    return jnp.mean(loss)


class Accuracy(hk.Module):
    def __call__(self, labels, preds):
        assert labels.shape == preds.shape[:-1]
        total = hk.get_state("total", (), dtype=jnp.int32, init=jnp.zeros) + labels.size
        nc = jnp.count_nonzero(labels == jnp.argmax(preds, axis=-1))
        num_correct = hk.get_state("num_correct", (), dtype=jnp.int32, init=jnp.zeros)
        num_correct = num_correct + nc
        hk.set_state("num_correct", num_correct)
        hk.set_state("total", total)
        return num_correct / total


def accuracy(labels, preds):
    return {"acc": Accuracy()(labels, preds)}


acc = hk.transform_with_state(accuracy)


@partial(jax.jit, static_argnums=(0, 3, 5))
def train_step(
    net_transform,
    params,
    net_state,
    optimizer,
    optimizer_state,
    metrics_transform,
    metrics_state,
    rng,
    inputs,
    labels,
):
    def loss_fun(params, state, rng, inputs, labels):
        preds, state = net_transform.apply(params, state, rng, inputs)
        loss = sparse_categorical_crossentropy_from_logits(labels, preds)
        return loss, (state, preds)

    (loss, (net_state, preds)), grad = jax.value_and_grad(loss_fun, has_aux=True)(
        params, net_state, rng, inputs, labels
    )
    updates, optimizer_state = optimizer.update(grad, optimizer_state, params)
    params = optax.apply_updates(params, updates)

    metrics, metrics_state = metrics_transform.apply(
        None, metrics_state, None, labels, preds
    )
    return loss, metrics, metrics_state, params, net_state, optimizer_state


@partial(jax.jit, static_argnums=(0, 3))
def test_step(
    net_transform,
    params,
    net_state,
    metrics_transform,
    metrics_state,
    rng,
    inputs,
    labels,
):
    preds, _ = net_transform.apply(params, net_state, rng, inputs)
    metrics, metrics_state = metrics_transform.apply(
        None, metrics_state, None, labels, preds
    )
    return metrics, metrics_state


def zeros_like(x):
    return jnp.zeros(
        tuple(1 if s is None else s for s in x.shape), x.dtype.as_numpy_dtype
    )


def fit(train_data, test_data, epochs=10, seed=0):
    rng = hk.PRNGSequence(seed)
    logger = tf.keras.callbacks.ProgbarLogger(
        count_mode="steps", stateful_metrics="acc"
    )
    logger.set_params(dict(verbose=True, epochs=epochs, steps=len(train_data)))
    net_transform = hk.transform_with_state(model_fn)
    inputs, labels = jax.tree_map(zeros_like, train_data.element_spec)
    net_params, net_state = net_transform.init(next(rng), inputs)
    preds, _ = net_transform.apply(net_params, net_state, next(rng), inputs)
    metrics_transform = hk.transform_with_state(accuracy)
    optimizer = optax.adam(1e-3)
    optimizer_state = optimizer.init(net_params)
    _, init_metrics_state = metrics_transform.init(next(rng), labels, preds)

    logger.on_train_begin({})
    for epoch in range(epochs):
        metrics_state = init_metrics_state
        logger.on_epoch_begin(epoch, {})
        # train
        for batch, example in enumerate(train_data):
            logger.on_train_batch_begin(batch)
            example = jax.tree_map(lambda x: jnp.asarray(x.numpy()), example)
            (
                loss,
                metrics,
                metrics_state,
                net_params,
                net_state,
                optimizer_state,
            ) = train_step(
                net_transform,
                net_params,
                net_state,
                optimizer,
                optimizer_state,
                metrics_transform,
                metrics_state,
                next(rng),
                *example,
            )
            # print("--")
            # print(jnp.mean(net_params["deq/lifted/conv2_d"]["w"]).to_py())
            metrics["loss"] = loss
            logs = {k: v.to_py() for k, v in metrics.items()}
            logger.on_train_batch_end(batch, logs)
        # validation
        metrics_state = init_metrics_state
        for batch, example in enumerate(test_data):
            logger.on_test_batch_begin(batch)
            example = jax.tree_map(lambda x: jnp.asarray(x.numpy()), example)
            metrics, metrics_state = test_step(
                net_transform,
                net_params,
                net_state,
                metrics_transform,
                metrics_state,
                next(rng),
                *example,
            )
        logs.update({f"val_{k}": v.to_py() for k, v in metrics.items()})
        logger.on_epoch_end(epoch, logs)


def main(_):
    def map_fn(image, label):
        image = image / 255 - 0.5
        return image, label

    train_ds, test_ds = tfds.load("mnist", split=("train", "test"), as_supervised=True)
    train_ds = train_ds.shuffle(1024).batch(batch_size).map(map_fn)
    test_ds = test_ds.batch(batch_size).map(map_fn)
    fit(train_ds, test_ds)


if __name__ == "__main__":
    app.run(main)
