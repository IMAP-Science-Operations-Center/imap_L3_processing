import json
from dataclasses import fields
from pathlib import Path
from typing import Type, T
from unittest.mock import Mock

import numpy as np

import tests
from imap_processing.swe.l3.models import SweConfiguration


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / "test_data" / filename


def get_test_instrument_team_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent.parent / "instrument_team_data" / filename


def build_swe_configuration(**args) -> SweConfiguration:
    with open(get_test_data_path("swe/example_swe_config.json")) as f:
        default_config = json.load(f)
    default_config.update(**args)
    return default_config


def create_dataclass_mock(obj: Type[T], **kwargs) -> T:
    return Mock(spec=[field.name for field in fields(obj)], **kwargs)


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
