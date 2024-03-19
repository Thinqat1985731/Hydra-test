# Standard Library
from dataclasses import dataclass

# Third Party Library
import hydra
import matplotlib.pyplot as plt

# First Party Library
from data import load_mnist
from neuralnet import NeuralNetMLP


@dataclass
class Experiment:
    exp_name: str
    n_hidden: int
    l2: float = 0.0
    epochs: int = 100
    eta: float = 0.001
    shuffle: bool = True
    minibatch_size: int = 1
    seed: int | None = None


@hydra.main(config_path="../conf", version_base=None, config_name="exp001")
def main(cfg: Experiment):
    exp_name = cfg.exp_name
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
    plt.savefig("outputs/" + exp_name + ".png")


if __name__ == "__main__":
    main()
