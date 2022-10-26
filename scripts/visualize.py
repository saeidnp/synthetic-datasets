from argparse import ArgumentParser
from synthetic_datasets import toy_2d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=100000)
    opts = parser.parse_args()

    dataset = toy_2d.get_dataset(opts.dataset, n_samples=opts.n_samples)
    data = np.array([x for x in dataset])
    fig, ax = plt.subplots()
    ax.hist2d(data[:, 0], data[:, 1], range=[[-1, 1], [-1, 1]], bins=200)

    fig.tight_layout()
    plt.show()