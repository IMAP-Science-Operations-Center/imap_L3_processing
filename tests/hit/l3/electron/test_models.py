from datetime import datetime, timedelta
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from spacepy.pycdf import CDF

from imap_l3_processing.cdf.imap_attribute_manager import ImapAttributeManager
from imap_l3_processing.hit.l3.electron.models import HitElectronDataProduct, HitL1bIalirtRates
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


def _build_electron_data_product() -> HitElectronDataProduct:
    return HitElectronDataProduct(
        input_metadata=Mock(),
        epoch=np.array([datetime(2026, 6, 1), datetime(2026, 6, 1, 0, 1)]),
        energy=np.arange(6, dtype=float),
        energy_delta_plus=np.arange(10, 16, dtype=float),
        energy_delta_minus=np.arange(20, 26, dtype=float),
        energy_labels=[
            "<300 keV",
            "300-400 keV",
            "400-800 keV",
            "800-1200 keV",
            "1200-2000 keV",
            ">2000 keV",
        ],
        a_side=np.full((2, 6), 1.0),
        a_side_delta_plus=np.full((2, 6), 2.0),
        a_side_delta_minus=np.full((2, 6), 3.0),
        b_side=np.full((2, 6), 4.0),
        b_side_delta_plus=np.full((2, 6), 5.0),
        b_side_delta_minus=np.full((2, 6), 6.0),
        hit_flags=np.array([0, 4], dtype=np.uint16),
    )


class TestHitElectronDataProduct(TestCase):
    def test_to_data_product_variables_emits_every_electron_variable_in_order(self) -> None:
        product = _build_electron_data_product()

        result = product.to_data_product_variables()

        expected = [
            ("epoch", product.epoch),
            ("epoch_delta", np.full(2, 30_000_000_000)),
            ("electron_energy", product.energy),
            ("electron_energy_delta_plus", product.energy_delta_plus),
            ("electron_energy_delta_minus", product.energy_delta_minus),
            ("electron_energy_label", product.energy_labels),
            ("electron_intensity_a_side", product.a_side),
            ("electron_intensity_a_side_delta_plus", product.a_side_delta_plus),
            ("electron_intensity_a_side_delta_minus", product.a_side_delta_minus),
            ("electron_intensity_b_side", product.b_side),
            ("electron_intensity_b_side_delta_plus", product.b_side_delta_plus),
            ("electron_intensity_b_side_delta_minus", product.b_side_delta_minus),
            ("hit_flags", product.hit_flags),
        ]
        self.assertEqual(13, len(result))
        for variable, (name, value) in zip(result, expected, strict=True):
            self.assertEqual(name, variable.name)
            np.testing.assert_array_equal(value, variable.value)

    def test_attribute_manager_resolves_every_emitted_variable_and_global_attrs(self) -> None:
        manager = ImapAttributeManager()
        manager.add_instrument_attrs("hit", "l3", "electron")

        with self.assertNoLogs(level="WARNING"):
            for variable in _build_electron_data_product().to_data_product_variables():
                with self.subTest(variable.name):
                    manager.get_variable_attributes(variable.name)

        self.assertEqual(
            "IMAP Mission HIT Instrument Level-3 Electron Intensity Science Data",
            manager.get_global_attributes("imap_hit_l3_electron")["Logical_source_description"],
        )
