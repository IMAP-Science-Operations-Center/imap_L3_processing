import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SSD_SHEET_NAME_FORMAT = "ssd(\d+)"

@dataclass
class CodiceHiMassCorrectionLookupTable:
    correction_factors: dict[int, np.ndarray]

    @classmethod
    def from_file(cls, file_path: Path):
        data = {}

        file = pd.read_excel(file_path, sheet_name=None, index_col=0, skiprows=[0])
        for sheet_name in file.keys():
            ssd_name_match = re.match(SSD_SHEET_NAME_FORMAT, sheet_name)
            if ssd_name_match:
                ssd = int(ssd_name_match.group(1))
                data[ssd] = file[sheet_name].to_numpy()
            else:
                logging.warning(f"Not using sheet {sheet_name} from {file_path.name} ancillary! Expected sheet names in the format: {SSD_SHEET_NAME_FORMAT}")

        assert len(data) > 0, f"Failed to read any correction factors from: {file_path.name}"

        return cls(correction_factors=data)

    def lookup_correction_factor(self, ssd_id: int, energy_channel: int, tof: int) -> float:
        data_for_ssd = self.correction_factors.get(ssd_id)
        if data_for_ssd is not None:
            return data_for_ssd[energy_channel, tof]
        else:
            return 1.0