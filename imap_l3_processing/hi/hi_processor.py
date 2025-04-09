from dataclasses import dataclass

import numpy as np
from imap_processing.spice.geometry import SpiceFrame

from imap_l3_processing import spice_wrapper
from imap_l3_processing.hi.l3.hi_l3_spectral_fit_dependencies import HiL3SpectralFitDependencies, \
    HI_L3_SPECTRAL_FIT_DESCRIPTOR
from imap_l3_processing.hi.l3.hi_l3_survival_dependencies import HiL3SurvivalDependencies
from imap_l3_processing.hi.l3.models import HiL3SpectralIndexDataProduct, GlowsL3eData, HiL1cData, \
    HiL3SurvivalCorrectedDataProduct
from imap_l3_processing.hi.l3.science.spectral_fit import spectral_fit
from imap_l3_processing.hi.l3.science.survival_probability import HiSurvivalProbabilityPointingSet, \
    HiSurvivalProbabilitySkyMap, Sensor
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class HiProcessor(Processor):
    def process(self):
        if "survival" in self.input_metadata.descriptor:
            hi_l3_survival_probabilities_dependencies = HiL3SurvivalDependencies.fetch_dependencies(self.dependencies)
            data_product = self._process_survival_probabilities(hi_l3_survival_probabilities_dependencies)
            save_data(data_product)
        else:
            hi_l3_spectral_fit_dependencies = HiL3SpectralFitDependencies.fetch_dependencies(self.dependencies)
            data_product = self._process_spectral_fit_index(hi_l3_spectral_fit_dependencies)
            save_data(data_product)

    def _process_spectral_fit_index(self, hi_l3_spectral_fit_dependencies):
        hi_l3_data = hi_l3_spectral_fit_dependencies.hi_l3_data

        epochs = hi_l3_data.epoch
        energy = hi_l3_data.energy
        fluxes = hi_l3_data.flux
        lons = hi_l3_data.lon
        lats = hi_l3_data.lat
        variances = hi_l3_data.variance

        gammas, errors = spectral_fit(len(epochs), len(lons), len(lats), fluxes, variances, energy)

        data_product = HiL3SpectralIndexDataProduct(
            energy=hi_l3_spectral_fit_dependencies.hi_l3_data.energy,
            epoch=hi_l3_spectral_fit_dependencies.hi_l3_data.epoch,
            spectral_fit_index_error=errors,
            spectral_fit_index=gammas,
            variance=hi_l3_spectral_fit_dependencies.hi_l3_data.variance,
            epoch_delta=hi_l3_spectral_fit_dependencies.hi_l3_data.epoch_delta,
            energy_deltas=hi_l3_spectral_fit_dependencies.hi_l3_data.energy_deltas,
            lat=hi_l3_spectral_fit_dependencies.hi_l3_data.lat,
            lon=hi_l3_spectral_fit_dependencies.hi_l3_data.lon,
            exposure=hi_l3_spectral_fit_dependencies.hi_l3_data.exposure,
            sensitivity=hi_l3_spectral_fit_dependencies.hi_l3_data.sensitivity,
            input_metadata=self.input_metadata.to_upstream_data_dependency(HI_L3_SPECTRAL_FIT_DESCRIPTOR),
            counts_uncertainty=hi_l3_spectral_fit_dependencies.hi_l3_data.counts_uncertainty,
            flux=hi_l3_spectral_fit_dependencies.hi_l3_data.flux,
            counts=hi_l3_spectral_fit_dependencies.hi_l3_data.counts,
        )

        return data_product

    def _process_survival_probabilities(self, hi_survival_probabilities_dependencies: HiL3SurvivalDependencies):
        spice_wrapper.furnish()
        combined_glows_hi = combine_glows_l3e_hi_l1c(hi_survival_probabilities_dependencies.glows_l3e_data,
                                                     hi_survival_probabilities_dependencies.hi_l1c_data)
        map_descriptor = parse_map_descriptor(self.input_metadata.descriptor)
        pointing_sets = []
        for hi_l1c, glows_l3e in combined_glows_hi:
            pointing_sets.append(HiSurvivalProbabilityPointingSet(hi_l1c, map_descriptor.sensor, glows_l3e))

        hi_survival_sky_map = HiSurvivalProbabilitySkyMap(pointing_sets, map_descriptor.grid_size,
                                                          SpiceFrame.ECLIPJ2000)

        survival_dataset = hi_survival_sky_map.to_dataset()

        data_product = HiL3SurvivalCorrectedDataProduct(
            input_metadata=self.input_metadata.to_upstream_data_dependency(self.input_metadata.descriptor),
            epoch=hi_survival_probabilities_dependencies.l2_data.epoch,
            energy=hi_survival_probabilities_dependencies.l2_data.energy,
            energy_deltas=hi_survival_probabilities_dependencies.l2_data.energy_deltas,
            counts=hi_survival_probabilities_dependencies.l2_data.counts,
            counts_uncertainty=hi_survival_probabilities_dependencies.l2_data.counts_uncertainty,
            epoch_delta=hi_survival_probabilities_dependencies.l2_data.epoch_delta,
            exposure=hi_survival_probabilities_dependencies.l2_data.exposure,
            flux=hi_survival_probabilities_dependencies.l2_data.flux / survival_dataset[
                "exposure_weighted_survival_probabilities"].values,
            lat=hi_survival_probabilities_dependencies.l2_data.lat,
            lon=hi_survival_probabilities_dependencies.l2_data.lon,
            sensitivity=hi_survival_probabilities_dependencies.l2_data.sensitivity,
            variance=hi_survival_probabilities_dependencies.l2_data.variance,
        )

        return data_product


@dataclass
class MapDescriptorParts:
    sensor: Sensor
    grid_size: int


def parse_map_descriptor(descriptor: str) -> MapDescriptorParts:
    sensor = Sensor(descriptor[:2])
    grid_size = int(descriptor.split("-")[4][0])

    return MapDescriptorParts(sensor, grid_size)


def combine_glows_l3e_hi_l1c(glows_l3e_data: list[GlowsL3eData], hi_l1c_data: list[HiL1cData]) -> list[
    tuple[HiL1cData, GlowsL3eData]]:
    l1c_by_epoch = {l1c.epoch: l1c for l1c in hi_l1c_data}
    glows_by_epoch = {l3e.epoch: l3e for l3e in glows_l3e_data}

    epochs = sorted(set(l1c_by_epoch.keys()).intersection(set(glows_by_epoch.keys())))

    return [(l1c_by_epoch[epoch], glows_by_epoch[epoch]) for epoch in epochs]
