from pathlib import Path

import numpy as np

import tests


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / filename

class NumpyArrayMatcher:
    def __init__(self, array, equal_nan=True):
        self.equal_nan = equal_nan
        self.array = array

    def __eq__(self, other):
        return np.array_equal(self.array, other, equal_nan=self.equal_nan)

    def __repr__(self):
        return repr(self.array)