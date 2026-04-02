
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        jnp.array(X_train),
        jnp.array(X_test),
        jnp.array(y_train),
        jnp.array(y_test),
    )

X_train,X_test,y_train,y_test = load_data() 
plt.plot(x_train)
plt.titile("Line plot of the X_data")
plt.show()

def model_apply(params, X):
    w, b = params
    logits = jnp.dot(X, w) + b
    return 1 / (1 + jnp.exp(-logits))  # sigmoid



def loss_fn(params, batch):
    X, y = batch

    preds = model_apply(params, X)

    eps = 1e-8
    loss = -jnp.mean(
        y * jnp.log(preds + eps) +
        (1 - y) * jnp.log(1 - preds + eps)
    )

    return loss



def accuracy(params, X, y):
    probs = model_apply(params, X)
    preds = (probs > 0.5).astype(jnp.int32)
    return jnp.mean(preds == y)


def sgd_update(params, grads, lr):
    w, b = params
    gw, gb = grads
    new_w = w - lr * gw
    new_b = b - lr * gb
    return (new_w, new_b)


@jit
def train_step(params, batch, lr):
    loss, grads = value_and_grad(loss_fn)(params, batch)
    new_params = sgd_update(params, grads, lr)
    return new_params, loss

def init_params(input_dim):
    key = jax.random.PRNGKey(0)

    w = jax.random.normal(key, (input_dim,))
    b = 0.0

    return (w, b)

def train():
    X_train, X_test, y_train, y_test = load_data()
    params = init_params(X_train.shape[1])
    lr = 0.01
    epochs = 200
    train_batch = (X_train, y_train)
    print("\nStarting Training...\n")

    for epoch in range(epochs):

        params, loss = train_step(params, train_batch, lr)

        if epoch % 20 == 0:
            train_acc = accuracy(params, X_train, y_train)
            test_acc = accuracy(params, X_test, y_test)

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Test Acc {test_acc:.4f}"
            )

    print("\nTraining Finished.")

    final_test_acc = accuracy(params, X_test, y_test)
    print(f"\nFinal Test Accuracy: {final_test_acc:.4f}")


if __name__ == "__main__":
    train()





