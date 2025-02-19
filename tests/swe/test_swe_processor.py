import unittest
from datetime import datetime
from unittest import skip
from unittest.mock import Mock

import numpy as np

from imap_processing.models import MagL1dData
from imap_processing.swapi.l3a.models import SwapiL3ProtonSolarWindData
from imap_processing.swe.l3.models import SweL2Data
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_processing.swe.swe_processor import SweProcessor


class TestSweProcessor(unittest.TestCase):
    @skip
    def test_something(self):
        start_date = datetime.now() - ti
        swe_l2_data = SweL2Data(
            epoch=np.array([]),
            epoch_delta=np.array([]),
            phase_space_density=np.array([]),
            flux=np.array([]),
            energy=np.array([]),
            inst_el=np.array([]),
            inst_az_spin_sector=np.array([])
        )

        mag_l1d_data = MagL1dData(
            epoch=np.array([]),
            mag_data=np.array([])
        )

        swapi_l3a_protion_data = SwapiL3ProtonSolarWindData(
            epoch=np.array([]),
            proton_sw_speed=np.array([]),
            proton_sw_density=np.array([]),
            proton_sw_temperature=np.array([]),
            proton_sw_clock_angle=np.array([]),
            proton_sw_deflection_angle=np.array([]),
            input_metadata=np.array([])
        )

        swel3_dependency = SweL3Dependencies(swe_l2_data, mag_l1d_data, swapi_l3a_protion_data)

        swe_processor = SweProcessor(dependencies=[], input_metadata=Mock())
        swel3_data = swe_processor.process_l3(swel3_dependency)
