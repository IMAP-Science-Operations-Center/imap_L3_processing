import dataclasses
from abc import ABC, abstractmethod

import numpy as np

from imap_l3_processing.data_utils import safe_divide
from imap_l3_processing.maps.map_models import IntensityMapData, HealPixIntensityMapData, RectangularIntensityMapData, \
    calculate_datetime_weighted_average


class CombinationStrategy(ABC):
    def combine_rectangular_intensity_map_data(self,
                                               maps: list[RectangularIntensityMapData]) -> RectangularIntensityMapData:
        for m in maps[1:]:
            assert np.array_equal(maps[0].coords.latitude_delta, m.coords.latitude_delta)
            assert np.array_equal(maps[0].coords.longitude_delta, m.coords.longitude_delta)
            assert np.array_equal(maps[0].coords.latitude_label, m.coords.latitude_label)
            assert np.array_equal(maps[0].coords.longitude_label, m.coords.longitude_label)

        self._check_maps_match([m.intensity_map_data for m in maps])
        intensity_map_data = self._combine([m.intensity_map_data for m in maps])
        return RectangularIntensityMapData(intensity_map_data=intensity_map_data, coords=maps[0].coords)

    def combine_healpix_intensity_map_data(self, maps: list[HealPixIntensityMapData]) -> HealPixIntensityMapData:
        for m in maps[1:]:
            assert np.array_equal(maps[0].coords.pixel_index, m.coords.pixel_index)
            assert np.array_equal(maps[0].coords.pixel_index_label, m.coords.pixel_index_label)

        self._check_maps_match([m.intensity_map_data for m in maps])
        intensity_map_data = self._combine([m.intensity_map_data for m in maps])
        return HealPixIntensityMapData(intensity_map_data=intensity_map_data, coords=maps[0].coords)

    @abstractmethod
    def _combine(self, maps: list[IntensityMapData]) -> IntensityMapData:
        raise NotImplementedError

    @staticmethod
    def _check_maps_match(maps: list[IntensityMapData]):
        first_map = maps[0]

        first_map_dict = dataclasses.asdict(first_map)

        fields_which_may_differ = {"ena_intensity", "ena_intensity_stat_uncert", "ena_intensity_sys_err",
                                   "bg_intensity", "bg_intensity_stat_uncert", "bg_intensity_sys_err",
                                   "exposure_factor", "obs_date", "obs_date_range"}

        differing_fields = []
        for field in dataclasses.fields(first_map):
            if field.name not in fields_which_may_differ:
                differing_fields.append(field.name)
                supports_nan = np.issubdtype(first_map_dict[field.name].dtype, np.floating)
                assert np.all(
                    [np.array_equal(dataclasses.asdict(m)[field.name], first_map_dict[field.name],
                                    equal_nan=supports_nan)
                     for m in maps]), f"{field.name}"

    @staticmethod
    def calculated_weighted_uncertainty(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        masked_values = np.where(weights == 0, 0, values)
        numerator = np.sum(np.square(masked_values * weights), axis=0)
        return np.sqrt(safe_divide(numerator, np.square(np.sum(weights, axis=0))))

    @staticmethod
    def calculate_weighted_sys_err(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        numerator = np.sum(values * weights, axis=0)
        denominator = np.sum(weights, axis=0)

        return safe_divide(numerator, denominator)


class UnweightedCombination(CombinationStrategy):
    def _combine(self, maps: list[IntensityMapData]) -> IntensityMapData:
        intensities = np.array([m.ena_intensity for m in maps])
        intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
        intensity_stat_unc = np.array([m.ena_intensity_stat_uncert for m in maps])
        exposures = np.array([m.exposure_factor for m in maps])

        mask = np.isnan(intensities) | (exposures == 0) | np.isnan(exposures)

        intensities = np.where(mask, 0, intensities)
        intensity_sys_err = np.where(mask, 0, intensity_sys_err)
        intensity_stat_unc = np.where(mask, 0, intensity_stat_unc)

        masked_exposures = np.where(mask, 0, exposures)
        summed_exposures = np.sum(masked_exposures, axis=0)
        weights = np.full_like(exposures, 1)
        masked_weights = np.where(mask, 0, weights)

        combined_intensity_stat_uncert = self.calculated_weighted_uncertainty(intensity_stat_unc, masked_weights)
        combined_intensity_sys_err = self.calculated_weighted_uncertainty(intensity_sys_err, masked_weights)

        summed_intensity = np.sum(intensities * masked_weights, axis=0)
        exposure_weighted_summed_intensity = safe_divide(summed_intensity, np.sum(masked_weights, axis=0))

        avg_obs_date = calculate_datetime_weighted_average(np.ma.array([m.obs_date for m in maps]), masked_weights,
                                                           0)

        return dataclasses.replace(maps[0],
                                   ena_intensity=exposure_weighted_summed_intensity,
                                   exposure_factor=summed_exposures,
                                   ena_intensity_sys_err=combined_intensity_sys_err,
                                   ena_intensity_stat_uncert=combined_intensity_stat_uncert,
                                   obs_date=avg_obs_date
                                   )


class ExposureWeightedCombination(CombinationStrategy):
    def _combine(self, maps: list[IntensityMapData]):
        intensities = np.array([m.ena_intensity for m in maps])
        intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
        intensity_stat_unc = np.array([m.ena_intensity_stat_uncert for m in maps])
        exposures = np.array([m.exposure_factor for m in maps])

        mask = np.isnan(intensities) | (exposures == 0) | np.isnan(exposures)

        intensities = np.where(mask, 0, intensities)
        intensity_sys_err = np.where(mask, 0, intensity_sys_err)
        intensity_stat_unc = np.where(mask, 0, intensity_stat_unc)

        masked_exposures = np.where(mask, 0, exposures)
        summed_exposures = np.sum(masked_exposures, axis=0)

        combined_intensity_stat_unc = self.calculated_weighted_uncertainty(intensity_stat_unc, masked_exposures)
        combined_intensity_sys_err = self.calculate_weighted_sys_err(intensity_sys_err, masked_exposures)

        summed_intensity = np.sum(intensities * masked_exposures, axis=0)
        exposure_weighted_summed_intensity = safe_divide(summed_intensity, np.sum(masked_exposures, axis=0))

        avg_obs_date = calculate_datetime_weighted_average(np.ma.array([m.obs_date for m in maps]), masked_exposures, 0)

        return dataclasses.replace(maps[0],
                                   ena_intensity=exposure_weighted_summed_intensity,
                                   exposure_factor=summed_exposures,
                                   ena_intensity_sys_err=combined_intensity_sys_err,
                                   ena_intensity_stat_uncert=combined_intensity_stat_unc,
                                   obs_date=avg_obs_date
                                   )


class UncertaintyWeightedCombination(CombinationStrategy):
    def _combine(self, maps: list[IntensityMapData]) -> IntensityMapData:
        intensities = np.array([m.ena_intensity for m in maps])
        intensity_sys_err = np.array([m.ena_intensity_sys_err for m in maps])
        intensity_stat_unc = np.array([m.ena_intensity_stat_uncert for m in maps])
        exposures = np.array([m.exposure_factor for m in maps])

        mask = np.isnan(intensities) | (exposures == 0) | np.isnan(exposures) | np.isnan(intensity_stat_unc)

        intensities = np.where(mask, 0, intensities)
        intensity_sys_err = np.where(mask, 0, intensity_sys_err)
        intensity_stat_unc = np.where(mask, 0, intensity_stat_unc)
        masked_exposures = np.where(mask, 0, exposures)
        summed_exposures = np.sum(masked_exposures, axis=0)

        squared_stat_unc = np.square(intensity_stat_unc)
        inverse_squared_stat_unc = safe_divide(np.ones_like(squared_stat_unc), squared_stat_unc)
        uncertainty_weighted_combined_intensity = safe_divide(
            np.nansum(safe_divide(intensities, squared_stat_unc), axis=0),
            np.nansum(inverse_squared_stat_unc, axis=0))
        combined_intensity_stat_unc = safe_divide(np.ones_like(inverse_squared_stat_unc[0]),
                                                  np.nansum(inverse_squared_stat_unc, axis=0))
        combined_intensity_sys_err = self.calculate_weighted_sys_err(intensity_sys_err, masked_exposures)

        avg_obs_date = calculate_datetime_weighted_average(np.ma.array([m.obs_date for m in maps]), masked_exposures, 0)

        return dataclasses.replace(maps[0],
                                   ena_intensity=uncertainty_weighted_combined_intensity,
                                   exposure_factor=summed_exposures,
                                   ena_intensity_sys_err=combined_intensity_sys_err,
                                   ena_intensity_stat_uncert=np.sqrt(combined_intensity_stat_unc),
                                   obs_date=avg_obs_date
                                   )
