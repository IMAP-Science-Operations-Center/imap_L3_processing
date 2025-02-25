import numpy as np
from imap_data_access import upload

from imap_processing.data_utils import rebin
from imap_processing.processor import Processor
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_solar_wind_velocity_vector
from imap_processing.swe.l3.models import SweL3Data
from imap_processing.swe.l3.science.pitch_calculations import average_flux, find_breakpoints, correct_and_rebin
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies
from imap_processing.utils import save_data


class SweProcessor(Processor):
    def process(self):
        dependencies = SweL3Dependencies.fetch_dependencies(self.dependencies)
        output_data = self.calculate_pitch_angle_products(dependencies)
        output_cdf = save_data(output_data)
        upload(output_cdf)

    def calculate_pitch_angle_products(self, dependencies: SweL3Dependencies) -> SweL3Data:
        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
        swe_epoch_delta = swe_l2_data.epoch_delta
        rebinned_mag = dependencies.mag_l1d_data.rebin_to(swe_epoch, swe_epoch_delta)

        swapi_l_a_proton_data = dependencies.swapi_l3a_proton_data
        swapi_epoch = swapi_l_a_proton_data.epoch
        solar_wind_vectors = calculate_solar_wind_velocity_vector(swapi_l_a_proton_data.proton_sw_speed,
                                                                  swapi_l_a_proton_data.proton_sw_clock_angle,
                                                                  swapi_l_a_proton_data.proton_sw_deflection_angle)

        rebinned_solar_wind_vectors = rebin(from_epoch=swapi_epoch,
                                            from_data=solar_wind_vectors,
                                            to_epoch=swe_epoch,
                                            to_epoch_delta=swe_epoch_delta)

        flux_by_pitch_angles = []
        for i in range(len(swe_epoch)):
            averaged_flux = average_flux(swe_l2_data.flux[i],
                                         np.array(dependencies.configuration["geometric_fractions"]))
            spacecraft_potential, halo_core = find_breakpoints(swe_l2_data.energy, averaged_flux)

            corrected_energy_bins = swe_l2_data.energy - spacecraft_potential
            rebinned_flux = correct_and_rebin(swe_l2_data.flux[i], corrected_energy_bins, swe_l2_data.inst_el,
                                              swe_l2_data.inst_az_spin_sector[i],
                                              rebinned_mag[i],
                                              rebinned_solar_wind_vectors[i],
                                              dependencies.configuration,
                                              )
            flux_by_pitch_angles.append(rebinned_flux)

        swe_l3_data = SweL3Data(input_metadata=self.input_metadata.to_upstream_data_dependency("sci"),
                                epoch=swe_epoch,
                                epoch_delta=swe_epoch_delta,
                                energy=dependencies.configuration["energy_bins"],
                                energy_delta_plus=dependencies.configuration["energy_delta_plus"],
                                energy_delta_minus=dependencies.configuration["energy_delta_minus"],
                                pitch_angle=dependencies.configuration["pitch_angle_bins"],
                                pitch_angle_delta=dependencies.configuration["pitch_angle_delta"],
                                flux_by_pitch_angle=np.array(flux_by_pitch_angles))
        return swe_l3_data
