import unittest
from unittest.mock import sentinel, Mock

from imap_l3_processing import map_models
from imap_l3_processing.map_models import RectangularCoords, SpectralIndexMapData, RectangularSpectralIndexMapData, \
    RectangularSpectralIndexDataProduct
from imap_l3_processing.models import DataProductVariable


class TestMapModels(unittest.TestCase):

    def test_rectangular_spectral_index_to_data_product_variables(self):
        input_metadata = Mock()

        spectral_index_data_product = RectangularSpectralIndexDataProduct(
            input_metadata=input_metadata,
            data=RectangularSpectralIndexMapData(
                spectral_index_map_data=SpectralIndexMapData(
                    epoch=sentinel.epoch,
                    epoch_delta=sentinel.epoch_delta,
                    energy=sentinel.energy,
                    energy_delta_plus=sentinel.energy_delta_plus,
                    energy_delta_minus=sentinel.energy_delta_minus,
                    energy_label=sentinel.energy_label,
                    latitude=sentinel.latitude,
                    longitude=sentinel.longitude,
                    exposure_factor=sentinel.exposure_factor,
                    obs_date=sentinel.obs_date,
                    obs_date_range=sentinel.obs_date_range,
                    solid_angle=sentinel.solid_angle,
                    ena_spectral_index=sentinel.ena_spectral_index,
                    ena_spectral_index_stat_unc=sentinel.ena_spectral_index_stat_unc
                ),
                coords=RectangularCoords(
                    latitude_delta=sentinel.latitude_delta,
                    latitude_label=sentinel.latitude_label,
                    longitude_delta=sentinel.longitude_delta,
                    longitude_label=sentinel.longitude_label,
                )
            )
        )

        actual_variables = spectral_index_data_product.to_data_product_variables()

        expected_variables = [
            DataProductVariable(map_models.EPOCH_VAR_NAME, sentinel.epoch),
            DataProductVariable(map_models.EPOCH_DELTA_VAR_NAME, sentinel.epoch_delta),
            DataProductVariable(map_models.ENERGY_VAR_NAME, sentinel.energy),
            DataProductVariable(map_models.ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
            DataProductVariable(map_models.ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
            DataProductVariable(map_models.ENERGY_LABEL_VAR_NAME, sentinel.energy_label),
            DataProductVariable(map_models.LATITUDE_VAR_NAME, sentinel.latitude),
            DataProductVariable(map_models.LATITUDE_DELTA_VAR_NAME, sentinel.latitude_delta),
            DataProductVariable(map_models.LATITUDE_LABEL_VAR_NAME, sentinel.latitude_label),
            DataProductVariable(map_models.LONGITUDE_VAR_NAME, sentinel.longitude),
            DataProductVariable(map_models.LONGITUDE_DELTA_VAR_NAME, sentinel.longitude_delta),
            DataProductVariable(map_models.LONGITUDE_LABEL_VAR_NAME, sentinel.longitude_label),
            DataProductVariable(map_models.EXPOSURE_FACTOR_VAR_NAME, sentinel.exposure_factor),
            DataProductVariable(map_models.OBS_DATE_VAR_NAME, sentinel.obs_date),
            DataProductVariable(map_models.OBS_DATE_RANGE_VAR_NAME, sentinel.obs_date_range),
            DataProductVariable(map_models.SOLID_ANGLE_VAR_NAME, sentinel.solid_angle),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_VAR_NAME, sentinel.ena_spectral_index),
            DataProductVariable(map_models.ENA_SPECTRAL_INDEX_STAT_UNC_VAR_NAME, sentinel.ena_spectral_index_stat_unc),
        ]

        self.assertEqual(expected_variables, actual_variables)


if __name__ == '__main__':
    unittest.main()
