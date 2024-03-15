# Standard Library
import os
import struct
from dataclasses import dataclass

# Third Party Library
import hydra
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# First Party Library
from neuralnet import NeuralNetMLP


def load_mnist(path: str, kind: str = "train") -> (npt.NDArray, npt.NDArray):
    labels_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        path,
        "%s-labels-idx1-ubyte" % kind,
    )
    images_path = os.path.join(
        os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
        path,
        "%s-images-idx3-ubyte" % kind,
    )

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.0) - 0.5) * 2  # ピクセル値の正規化

    return images, labels


@dataclass
class Experiment:
    n_hidden: int
    l2: float = 0.0
    epochs: int = 100
    eta: float = 0.001
    shuffle: bool = True
    minibatch_size: int = 1
    seed: int | None = None


@hydra.main(config_path="../conf", version_base=None, config_name="exp001")
def main(cfg: Experiment):
    n_hidden = cfg.n_hidden
    l2 = cfg.l2
    epochs = cfg.epochs
    eta = cfg.eta
    shuffle = cfg.shuffle
    minibatch_size = cfg.minibatch_size
    seed = cfg.seed

    X_train, y_train = load_mnist("data", kind="train")
    X_test, y_test = load_mnist("data", kind="t10k")
    print("Data Load Completed!")

    nn = NeuralNetMLP(n_hidden, l2, epochs, eta, minibatch_size, shuffle, seed)
    nn.fit(
        X_train=X_train[:55000],
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:],
    )

    plt.plot(range(nn.epochs), nn.eval_["cost"])
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.plot()
    plt.savefig("exp001_cost.png")


if __name__ == "__main__":
    main()
