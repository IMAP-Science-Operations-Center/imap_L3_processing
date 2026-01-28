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
from imap_l3_processing.maps.map_models import ISNRateData, ISNBackgroundSubtractedData, ISNBackgroundSubtractedMapData
from imap_l3_processing.maps.map_models import RectangularIntensityMapData, RectangularSpectralIndexDataProduct, \
    RectangularSpectralIndexMapData, RectangularIntensityDataProduct, InputRectangularPointingSet, \
    ISNBackgroundSubtractedDataProduct
from imap_l3_processing.maps.map_processor import MapProcessor
from imap_l3_processing.maps.spectral_fit import fit_spectral_index_map, slice_energy_range_by_bin
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
                spectral_fit_data = perform_spectral_fit(deps.map_data, descriptor.spectral_index_energy_range or (4, 8))
                data_product = RectangularSpectralIndexDataProduct(self.input_metadata, spectral_fit_data)
            case MapDescriptorParts(quantity=MapQuantity.SpectralIndexNBS):
                deps = LoL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
                spectral_fit_data = perform_spectral_fit(deps.map_data, descriptor.spectral_index_energy_range or (0, 8))
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
        return dataclasses.replace(pset, exposure_times=np.sum(pset.exposure_times, axis=3),
                                   hae_longitude=np.mean(pset.hae_longitude, axis=-1),
                                   hae_latitude=np.mean(pset.hae_latitude, axis=-1))


def perform_spectral_fit(data: RectangularIntensityMapData, spectral_index_range: tuple[int,int]) -> RectangularSpectralIndexMapData:
    intensity_data = slice_energy_range_by_bin(data.intensity_map_data, spectral_index_range[0], spectral_index_range[1])
    return RectangularSpectralIndexMapData(
        spectral_index_map_data=fit_spectral_index_map(intensity_data),
        coords=data.coords
    )


def isn_background_subtraction(isn_rate_data: ISNRateData) -> ISNBackgroundSubtractedMapData:
    isn_rate_background_subtracted = isn_rate_data.ena_count_rate - isn_rate_data.bg_rate
    isn_bg_subtracted_stat_err = np.sqrt(
        np.square(isn_rate_data.ena_count_rate_stat_uncert) + np.square(isn_rate_data.bg_rate_stat_uncert)
    )

    map_data = ISNBackgroundSubtractedData(
        epoch=isn_rate_data.epoch,
        epoch_delta=isn_rate_data.epoch_delta,
        energy_delta_plus=isn_rate_data.energy_delta_plus,
        energy_delta_minus=isn_rate_data.energy_delta_minus,
        energy_label=isn_rate_data.energy_label,
        obs_date=isn_rate_data.obs_date,
        obs_date_range=isn_rate_data.obs_date_range,
        ena_intensity=isn_rate_data.ena_intensity,
        ena_intensity_stat_uncert=isn_rate_data.ena_intensity_stat_uncert,
        ena_intensity_sys_err=isn_rate_data.ena_intensity_sys_err,
        energy=isn_rate_data.energy,
        exposure_factor=isn_rate_data.exposure_factor,
        solid_angle=isn_rate_data.solid_angle,
        bg_rate=isn_rate_data.bg_rate,
        bg_rate_stat_uncert=isn_rate_data.bg_rate_stat_uncert,
        bg_rate_sys_err=isn_rate_data.bg_rate_sys_err,
        ena_count_rate=isn_rate_data.ena_count_rate,
        ena_count_rate_stat_uncert=isn_rate_data.ena_count_rate_stat_uncert,
        latitude=isn_rate_data.latitude,
        longitude=isn_rate_data.longitude,
        isn_bg_rate_subtracted=isn_rate_background_subtracted,
        isn_bg_rate_subtracted_stat_uncert=isn_bg_subtracted_stat_err,
        isn_bg_rate_subtracted_sys_err=isn_rate_data.bg_rate_sys_err
    )

    return ISNBackgroundSubtractedMapData(isn_rate_map_data=map_data)
