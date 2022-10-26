from argparse import ArgumentParser
from synthetic_datasets import toy_2d
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    num_datasets = len(toy_2d.dataset_classes)
    nrows = int(np.ceil(np.sqrt(num_datasets)))
    ncols = int(np.ceil(num_datasets / nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i, dataset_name in enumerate(toy_2d.dataset_classes):
        dataset = toy_2d.get_dataset(dataset_name, n_samples=100000)
        data = np.array([x for x in dataset])
        axes[i].hist2d(data[:, 0], data[:, 1], range=[[-1, 1], [-1, 1]], bins=200)
        axes[i].set_title(dataset_name)

    for i in range(num_datasets, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    plt.show()