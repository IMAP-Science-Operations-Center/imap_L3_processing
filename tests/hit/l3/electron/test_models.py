from datetime import datetime, timedelta

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.hit.l3.electron.models import HitL1bIalirtRates
from tests.temp_file_test_case import TempFileTestCase

FILL_VALUE = np.float32(-9.223372e18)
EPOCH_COUNT = 3
IALIRT_RATE_COUNT = 20


class TestHitL1bIalirtRates(TempFileTestCase):
    def test_read_from_cdf_reads_ialirt_rates_and_replaces_fill_with_nan(self) -> None:
        epoch = np.array(
            [datetime(2026, 6, 1) + timedelta(seconds=60 * index) for index in range(EPOCH_COUNT)]
        )
        rates = np.arange(EPOCH_COUNT * IALIRT_RATE_COUNT, dtype=np.float32).reshape(
            EPOCH_COUNT, IALIRT_RATE_COUNT
        )
        rates_delta_plus = np.arange(
            100, 100 + EPOCH_COUNT * IALIRT_RATE_COUNT, dtype=np.float32
        ).reshape(EPOCH_COUNT, IALIRT_RATE_COUNT)
        rates_delta_minus = np.arange(
            200, 200 + EPOCH_COUNT * IALIRT_RATE_COUNT, dtype=np.float32
        ).reshape(EPOCH_COUNT, IALIRT_RATE_COUNT)
        rates[1, 2] = FILL_VALUE

        cdf_path = self.temp_directory / "imap_hit_l1b_standard-rates_20260601_v001.cdf"
        with CDF(str(cdf_path), "") as cdf:
            cdf["epoch"] = epoch
            cdf["ialirtrates"] = rates
            cdf["ialirtrates"].attrs["FILLVAL"] = FILL_VALUE
            cdf["ialirtrates_stat_uncert_plus"] = rates_delta_plus
            cdf["ialirtrates_stat_uncert_minus"] = rates_delta_minus

        result = HitL1bIalirtRates.read_from_cdf(cdf_path)

        expected_rates = rates.astype(float)
        expected_rates[1, 2] = np.nan
        expected_rates_delta_plus = rates_delta_plus.astype(float)
        expected_rates_delta_plus[1, 2] = np.nan
        expected_rates_delta_minus = rates_delta_minus.astype(float)
        expected_rates_delta_minus[1, 2] = np.nan
        filled_index = np.array([[1, 2]])

        np.testing.assert_array_equal(epoch, result.epoch)
        np.testing.assert_array_equal(expected_rates, result.rates)
        np.testing.assert_array_equal(expected_rates_delta_plus, result.rates_delta_plus)
        np.testing.assert_array_equal(expected_rates_delta_minus, result.rates_delta_minus)
        np.testing.assert_array_equal(filled_index, np.argwhere(np.isnan(result.rates)))
        np.testing.assert_array_equal(filled_index, np.argwhere(np.isnan(result.rates_delta_plus)))
        np.testing.assert_array_equal(
            filled_index, np.argwhere(np.isnan(result.rates_delta_minus))
        )
