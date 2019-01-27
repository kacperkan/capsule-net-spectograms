from collections import defaultdict, Counter
from typing import *

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

np.random.seed(0)


class TrainLoader(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, rebalance_samples: bool = True,
                 transforms: Optional = None):
        self._x = x_data.astype(np.float32)
        self._y = y_data.astype(np.float32)
        self._transforms = transforms

        if rebalance_samples:
            self.rebalance_samples()

    def __getitem__(self, index):
        x, y = self._x[index], self._y[index]
        if self._transforms is not None:
            x = self._transforms(x)
        return x[0].T, np.argmax(y)

    def __len__(self):
        return len(self._x)

    def rebalance_samples(self):
        classes = np.argmax(self._y, axis=1)
        counts = Counter(classes)
        max_count = np.max(list(counts.values()))
        unique_classes = counts.keys()
        new_xs = [self._x]
        new_ys = [self._y]
        for cls in unique_classes:
            samples_left = max_count - counts[cls]
            if samples_left > 0:
                available_indices = np.where(classes == cls)[0]
                indices = np.random.choice(available_indices, size=samples_left)
                new_xs.append(self._x[indices])
                new_ys.append(self._y[indices])
        self._x = np.concatenate(new_xs, axis=0)
        self._y = np.concatenate(new_ys, axis=0)


class TestLoader(Dataset):
    def __init__(self, x_path: str, transforms: Optional = None):
        self._x = np.load(x_path).astype(np.float32)
        self._transforms = transforms

    def __getitem__(self, index):
        sample = self._x[index]

        if self._transforms is not None:
            sample = self._transforms(sample)

        return sample[0].T

    def __len__(self):
        return len(self._x)


def generate_submission(predictions: list, output_path: str):
    data = defaultdict(list)
    for i, p in enumerate(predictions):
        data['id'].append(i + 1)
        data['class'].append(p)
    frame = pd.DataFrame(data=data)
    frame.to_csv(output_path, index=False)


def generate_submission_with_class_confidences(predictions: list, output_path: str):
    data = defaultdict(list)
    for i, p in enumerate(predictions):
        data['id'].append(i + 1)
        data['c1'].append(p[0])
        data['c2'].append(p[1])
        data['c3'].append(p[2])

    frame = pd.DataFrame(data=data)
    frame.to_csv(output_path, index=False)
