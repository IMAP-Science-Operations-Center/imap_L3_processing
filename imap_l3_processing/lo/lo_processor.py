import dataclasses
from pathlib import Path

import numpy as np
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing.lo.l3.lo_l3_isn_background_subtracted_dependencies import \
    LoL3ISNBackgroundSubtractedDependencies
from imap_l3_processing.lo.l3.lo_l3_spectral_fit_dependencies import LoL3SpectralFitDependencies
from imap_l3_processing.maps.hilo_l3_survival_dependencies import HiLoL3SurvivalDependencies
from imap_l3_processing.maps.map_descriptors import parse_map_descriptor, MapDescriptorParts, MapQuantity, \
    SurvivalCorrection, ReferenceFrame
from imap_l3_processing.maps.map_models import ISNRateData, ISNBackgroundSubtractedData
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData, RectangularIntensityDataProduct, InputRectangularPointingSet, \
    ISNBackgroundSubtractedDataProduct
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.spectral_fit import fit_spectral_index_map
from imap_l3_processing.maps.survival_probability_processing import process_survival_probabilities
from imap_l3_processing.models import Instrument
from imap_l3_processing.utils import save_data


class LoProcessor(MapProcessor):
    def process(self, spice_frame_name: SpiceFrame = SpiceFrame.ECLIPJ2000) -> list[Path]:
        set_of_parent_file_names = set(self.get_parent_file_names())
        descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        match descriptor:
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndex):
                deps = LoL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                spectral_fit_data = perform_spectral_fit(deps.map_data)
                data_product = RectangularSpectralIndexDataProduct(self.input_metadata, spectral_fit_data)
            case MapDescriptorParts(quantity=MapQuantity.ISNBackgroundSubtracted):
                deps = LoL3ISNBackgroundSubtractedDependencies.fetch_dependencies(self.dependencies)
                background_subtracted = isn_background_subtraction(deps.map_data)
                data_product = ISNBackgroundSubtractedDataProduct(self.input_metadata, background_subtracted)
            case MapDescriptorParts(survival_correction=SurvivalCorrection.SurvivalCorrected,
                                    reference_frame=ReferenceFrame.Spacecraft | ReferenceFrame.Heliospheric):
                deps = HiLoL3SurvivalDependencies.fetch_dependencies(self.dependencies, Instrument.IMAP_LO)
                deps.l1c_data = list(map(self._collapse_pset_dimension, deps.l1c_data))
                data = process_survival_probabilities(deps, spice_frame_name)

                data_product = RectangularIntensityDataProduct(self.input_metadata, data)
                set_of_parent_file_names.update(path.name for path in deps.dependency_file_paths)
            case None:
                raise ValueError(f"Could not parse descriptor {self.input_metadata.descriptor}")
            case _:
                raise NotImplementedError(self.input_metadata.descriptor)

        data_product.parent_file_names = sorted(set_of_parent_file_names)
        return [save_data(data_product)]

    @staticmethod
    def _collapse_pset_dimension(pset: InputRectangularPointingSet) -> InputRectangularPointingSet:
        return dataclasses.replace(pset, exposure_times=np.sum(pset.exposure_times, axis=3))


def perform_spectral_fit(data: RectangularIntensityMapData) -> RectangularSpectralIndexMapData:
    return RectangularSpectralIndexMapData(
        spectral_index_map_data=fit_spectral_index_map(data.intensity_map_data),
        coords=data.coords
    )


def isn_background_subtraction(isn_rate_data: ISNRateData) -> ISNBackgroundSubtractedData:
    isn_rate_background_subtracted = isn_rate_data.ena_count_rate - isn_rate_data.bg_rates

    return ISNBackgroundSubtractedData(
        epoch=isn_rate_data.epoch,
        counts=isn_rate_data.counts,
        ena_intensity=isn_rate_data.ena_intensity,
        ena_intensity_stat_uncert=isn_rate_data.ena_intensity_stat_uncert,
        ena_intensity_sys_err=isn_rate_data.ena_intensity_sys_err,
        energy=isn_rate_data.energy,
        energy_stat_uncert=isn_rate_data.energy_stat_uncert,
        exposure_factor=isn_rate_data.exposure_factor,
        geometric_factor=isn_rate_data.geometric_factor,
        geometric_factor_stat_uncert=isn_rate_data.geometric_factor_stat_uncert,
        solid_angle=isn_rate_data.solid_angle,
        bg_rates=isn_rate_data.bg_rates,
        bg_rates_stat_uncert=isn_rate_data.bg_rates_stat_uncert,
        bg_rates_sys_err=isn_rate_data.bg_rates_sys_err,
        ena_count_rate=isn_rate_data.ena_count_rate,
        ena_count_rate_stat_uncert=isn_rate_data.ena_count_rate_stat_uncert,
        ena_count_rate_sys_uncert=np.zeros_like(isn_rate_data.ena_count_rate),
        # ena count rate systematic uncertainty
        latitude=isn_rate_data.latitude,
        longitude=isn_rate_data.longitude,
        isn_rate_background_subtracted=isn_rate_background_subtracted,
        bg_subtracted_stat_err=np.sqrt(np.square(isn_rate_data.ena_count_rate_stat_uncert) + np.square(
            isn_rate_data.bg_rates_stat_uncert)),
        bg_subtracted_sys_uncertainty=isn_rate_data.bg_rates_sys_err
    )
