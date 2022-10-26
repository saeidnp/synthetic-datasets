# This file heavily borrows from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
import sklearn.datasets
import numpy as np
from torch.utils.data import Dataset
from common_utils.random import RNG


class ToyDatasetBase(Dataset):
    def __init__(self, *args, **kwargs):
        self.setup_data(*args, **kwargs)

    def setup_data(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class SwissRoll(ToyDatasetBase):
    def setup_data(self, n_samples, noise=1.0):
        data = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=noise)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 20  # Makes the data fit in the unit square
        self.data = data


class Circle(ToyDatasetBase):
    def setup_data(self, n_samples, noise=0.08):
        data = sklearn.datasets.make_circles(n_samples=100000, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3 / 4  # Makes the data fit in the unit square
        self.data = data


class Rings(ToyDatasetBase):
    def setup_data(self, n_samples):
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [
                    np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * 3.0
        )
        np.random.shuffle(X)  # , random_state=rng)

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        data = X.astype("float32")
        data /= 4  # Makes the data fit in the unit square
        self.data = data


class Moons(ToyDatasetBase):
    def setup_data(self, n_samples):
        data = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        data /= 4
        self.data = data


class EightGaussians(ToyDatasetBase):
    def setup_data(self, n_samples, scale=1.0):
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]

        data = []
        for _ in range(n_samples):
            point = np.random.randn(2) / 8 * scale
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            data.append(point)
        data = np.array(data, dtype="float32")
        data /= 1.414
        self.data = data


class PinWheel(ToyDatasetBase):
    def setup_data(self, n_samples):
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes * num_per_class, 2) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        data /= 4
        self.data = data


class TwoSpirals(ToyDatasetBase):
    def setup_data(self, n_samples):
        n = np.sqrt(np.random.rand(n_samples // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(n_samples // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        self.data = x / 4


class Checkerboard(ToyDatasetBase):
    def setup_data(self, n_samples):
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) / 2
        self.data = data


class Line(ToyDatasetBase):
    def setup_data(self, n_samples):
        x = np.random.rand(n_samples) * 2 - 1
        y = x
        self.data = np.stack((x, y), 1)


class Sine(ToyDatasetBase):
    def setup_data(self, n_samples):
        x = np.random.rand(n_samples) * 2 - 1
        y = np.sin(np.pi * x)
        self.data = np.stack((x, y), 1)


dataset_classes = {
    "8gaussians": EightGaussians,
    "pinwheel": PinWheel,
    "2spirals": TwoSpirals,
    "checkerboard": Checkerboard,
    "line": Line,
    "sine": Sine,
    "swissroll": SwissRoll,
    "moons": Moons,
}


def get_dataset(dataset_name, n_samples, seed=123, **kwargs):
    dataset_name = dataset_name.lower()
    assert dataset_name in dataset_classes, f"Unknown dataset: {dataset_name}"
    with RNG(seed):
        return dataset_classes[dataset_name](n_samples, **kwargs)