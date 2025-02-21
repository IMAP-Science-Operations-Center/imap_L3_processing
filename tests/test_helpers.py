from pathlib import Path

import numpy as np

import tests
from imap_processing.swe.l3.swe_l3_dependencies import SweConfiguration


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / "test_data" / filename


def get_test_instrument_team_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent.parent / "instrument_team_data" / filename


def build_swe_configuration(**args):
    default_config = SweConfiguration(geometric_fractions=[], pitch_angle_bins=[], pitch_angle_delta=[], energy_bins=[],
                                      energy_delta_plus=[], energy_delta_minus=[])
    default_config.update(**args)
    return default_config


class NumpyArrayMatcher:
    def __init__(self, array, equal_nan=True):
        self.equal_nan = equal_nan
        self.array = array

    def __eq__(self, other):
        if isinstance(self.array, (np.ndarray, list)):
            return np.array_equal(self.array, other, equal_nan=self.equal_nan)
        else:
            return self.array == other

    def __repr__(self):
        return repr(self.array)
