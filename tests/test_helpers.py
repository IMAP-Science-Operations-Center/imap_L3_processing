import json
import logging
import os
from dataclasses import fields
from pathlib import Path
from typing import Type, T
from unittest.mock import Mock

import numpy as np
import spiceypy

import imap_l3_processing
import tests
from imap_l3_processing.swe.l3.models import SweConfiguration
from imap_l3_processing.swe.l3.science.moment_calculations import Moments, MomentFitResults


def get_run_local_data_path(extension: str) -> Path:
    return Path(tests.__file__).parent.parent / "run_local_input_data" / extension


def try_get_many_run_local_paths(extensions: list[str]) -> tuple[bool, list[Path]]:
    missing_path = False
    paths = []
    for extension in extensions:
        paths.append(get_run_local_data_path(extension))
        if not paths[-1].exists():
            missing_path = True
    return missing_path, paths


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
    def __init__(self, array, equal_nan=True, almost_equal=False):
        self.equal_nan = equal_nan
        self.array = array
        self.almost_equal = almost_equal

    def __eq__(self, other):
        if isinstance(self.array, (np.ndarray, list)):
            if not self.almost_equal:
                return np.array_equal(self.array, other, equal_nan=self.equal_nan)
            else:
                return np.allclose(self.array, other)
        else:
            return self.array == other

    def __repr__(self):
        return repr(self.array)


def assert_dict_close(x, y, rtol=1e-7, path=None):
    if path is None:
        path = []
    path_str = " > ".join(path)
    if isinstance(x, dict) and isinstance(y, dict):
        assert set(x.keys()) == set(
            y.keys()), f"keys differ at {path_str}\n expected keys: {x.keys()}\n actual keys: {y.keys()}"
        for k in x:
            assert_dict_close(x[k], y[k], rtol, path.copy() + [k])
    elif isinstance(x, (list, np.ndarray, float)):
        np.testing.assert_allclose(x, y, rtol=rtol, err_msg=f"path to failure: {path_str}")
    else:
        assert x == y, f"{x} != {y} at path {path_str}"


def assert_dataclass_fields(expected_obj, actual_obj, omit=None):
    omit = omit or []
    for field in [f for f in fields(actual_obj) if f not in omit]:
        expected = getattr(expected_obj, field.name)
        actual = getattr(actual_obj, field.name)
        if isinstance(actual, (list, np.ndarray, float)):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            assert_dict_close(expected, actual, rtol=1e-20)
        else:
            assert expected == actual, f"{expected} != {actual} for field {field.name}"


def environment_variables(env_vars: dict):
    def decorator(func):

        def wrapper(*args, **kwargs):
            old_vars = {k: os.environ.get(v) for k, v in env_vars.items() if os.environ.get(str(v)) is not None}
            for k, v in env_vars.items():
                os.environ[k] = str(v)

            func_result = func(*args, **kwargs)

            for k in env_vars.keys():
                del os.environ[k]

            for k, v in old_vars.items():
                os.environ[k] = v

            return func_result

        return wrapper

    return decorator


def furnish_local_spice():
    logger = logging.getLogger(__name__)

    kernels = Path(imap_l3_processing.__file__).parent.parent.joinpath("spice_kernels")
    for file in kernels.iterdir():
        logger.log(logging.INFO, f"loading packaged kernel: {file}")
        spiceypy.furnsh(str(file))
