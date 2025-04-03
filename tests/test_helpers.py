import json
from dataclasses import fields
from pathlib import Path
from typing import Type, T
from unittest.mock import Mock

import numpy as np

import tests
from imap_l3_processing.swe.l3.models import SweConfiguration
from imap_l3_processing.swe.l3.science.moment_calculations import Moments, MomentFitResults


def get_test_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent / "test_data" / filename


def get_test_data_folder() -> Path:
    return Path(tests.__file__).parent / "test_data"


def get_test_instrument_team_data_path(filename: str) -> Path:
    return Path(tests.__file__).parent.parent / "instrument_team_data" / filename


def build_swe_configuration(**args) -> SweConfiguration:
    with open(get_test_data_path("swe/example_swe_config.json")) as f:
        default_config = json.load(f)
    default_config.update(**args)
    return default_config


def build_moments(**args) -> Moments:
    default_moments = dict(alpha=1, beta=2, t_parallel=3e5, t_perpendicular=4e5, velocity_x=500, velocity_y=600,
                           velocity_z=700, density=80, aoo=9, ao=10)
    default_moments.update(**args)

    return Moments(**default_moments)


def build_moment_fit_results(moments: Moments = None, chisq: float = 1, number_of_points: int = 10,
                             regress_result: np.ndarray = None) -> MomentFitResults:
    if moments is None:
        moments = build_moments()
    if regress_result is None:
        regress_result = np.ndarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    return MomentFitResults(moments=moments, chisq=chisq, number_of_points=number_of_points,
                            regress_result=regress_result)


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


def assert_dict_close(x, y, rtol=1e-7, path=None):
    if path is None:
        path = []
    path_str = " > ".join(path)
    if isinstance(x, dict) and isinstance(y, dict):
        assert set(x.keys()) == set(y.keys()), f"keys differ at {path_str}"
        for k in x:
            assert_dict_close(x[k], y[k], rtol, path.copy() + [k])
    elif isinstance(x, (list, np.ndarray)):
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg=f"path to failure: {path_str}")
    else:
        assert x == y, f"{x} != {y} at path {path_str}"
