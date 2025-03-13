import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class BadAngleFlagConfiguration:
    mask_close_to_uv_source: bool
    mask_inside_excluded_region: bool
    mask_excluded_by_instr_team: bool
    mask_suspected_transient: bool

    def evaluate_flags(self, flags: np.ndarray):
        configuration = np.array([self.mask_close_to_uv_source,
                                  self.mask_inside_excluded_region,
                                  self.mask_excluded_by_instr_team,
                                  self.mask_suspected_transient]).reshape(4, 1)

        return np.any(np.logical_and(configuration, flags), axis=-2)

    @classmethod
    def from_file(cls, file_path: Union[Path, str]):
        with open(file_path) as f:
            config = json.load(f)

        return cls(**config)
