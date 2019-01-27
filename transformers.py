import random

import numpy as np

import constants


class Normalizer:
    def __call__(self, sample):
        # return (sample - constants.DATA_MEAN) / constants.DATA_STD
        return (sample - constants.DATA_MAX) / (constants.DATA_MAX - constants.DATA_MIN)


class RandomWhiteNoise:
    def __init__(self, min_value, max_value):
        self._min = min_value
        self._max = max_value

    def __call__(self, sample):
        noise = np.random.uniform(self._min, self._max)
        return sample + noise


class MultiplicativeNoise:
    def __init__(self, min_value, max_value):
        self._min = min_value
        self._max = max_value

    def __call__(self, sample):
        noise = np.random.uniform(self._min, self._max)
        return sample * noise


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, sample):
        if random.random() < self._p:
            sample = np.flip(sample, axis=1).copy()
        return sample


class RandomBrightness:
    def __init__(self, min_value: float, max_value: float):
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, sample):
        add = np.random.uniform(self._min_value, self._max_value)
        sample = sample + add
        sample = np.clip(sample, constants.DATA_MIN, constants.DATA_MAX)
        return sample


class RandomContrast:
    def __init__(self, min_value: float, max_value: float):
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, sample):
        mult = np.random.uniform(self._min_value, self._max_value)
        sample = sample * mult
        sample = np.clip(sample, constants.DATA_MIN, constants.DATA_MAX)
        return sample
