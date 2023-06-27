import tensorflow as tf
import matplotlib.pyplot as plt

from dpconvcnp.data.data import Batch


def plot_batch(batch: Batch):
    
    x_ctx = batch.x_ctx.numpy()[0]
    x_trg = batch.x_trg.numpy()[0]
    y_ctx = batch.y_ctx.numpy()[0]
    y_trg = batch.y_trg.numpy()[0]

    print(x_ctx.shape)
    print(x_trg.shape)
    print(y_ctx.shape)
    print(y_trg.shape)

    plt.scatter(x_ctx[:, 0], y_ctx[:, 0], c="red", label="x_ctx")
    plt.scatter(x_trg[:, 0], y_trg[:, 0], c="green", label="x_trg")
    plt.savefig("batch.png")