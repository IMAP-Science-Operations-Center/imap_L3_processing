from datetime import timedelta

import numpy as np

from imap_processing.data_utils import rebin
from imap_processing.processor import Processor
from imap_processing.swapi.l3a.science.calculate_pickup_ion import calculate_solar_wind_velocity_vector
from imap_processing.swe.l3.models import SweL3Data
from imap_processing.swe.l3.science.pitch_calculations import average_flux, find_breakpoints, correct_and_rebin
from imap_processing.swe.l3.swe_l3_dependencies import SweL3Dependencies


class SweProcessor(Processor):
    def process(self):
        pass

    def calculate_pitch_angle_products(self, dependencies: SweL3Dependencies) -> SweL3Data:
        # step 1 rebin

        swe_l2_data = dependencies.swe_l2_data
        swe_epoch = swe_l2_data.epoch
        swe_epoch_delta = np.full(swe_epoch.shape, timedelta(seconds=30))
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

        for i in range(len(swe_epoch)):
            averaged_flux = average_flux(swe_l2_data.flux[i],
                                         np.array(dependencies.configuration["geometric_fractions"]))
            spacecraft_potential, halo_core = find_breakpoints(swe_l2_data.energy, averaged_flux)

            corrected_energy_bins = swe_l2_data.energy - spacecraft_potential
            pitch_angle_bins, output_energies = [0, 90, 180], [1, 10, 100]
            correct_and_rebin(swe_l2_data.flux[i], corrected_energy_bins, swe_l2_data.inst_el,
                              swe_l2_data.inst_az_spin_sector[i],
                              rebinned_mag[i],
                              rebinned_solar_wind_vectors[i],
                              pitch_angle_bins,
                              output_energies,
                              )

        # iterate minute chucks of SweL3Data, mag vector, swapi sw vector
        #   flux = calulate average flux
        #   spacecraft_potential, halo_core = swapi_epochs(energies ,flux)
        #   (maybe using previous values as initial guess)
        #
        #   corrected_energies = energies - spacecraft_potential
        #   particle_velocity_despun = calculate_velocity_in_dsp_frame_km_s(
        #       corrected_energies, inst_el, inst_az)
        #   particle_velocity_sw_frame = calculate_velocity_in_sw_frame(
        #       particle_velocity_despun, solar_wind_velocity)
        #
        #   pitch_angle = calculate_pitch_angle(particle_velocity_sw_frame, mag_field_vector)
        #   gyrophase = calculate_gyrophase(particle_velocity_sw_frame, mag_field_vector)
        #   energy_in_sw_frame = calculate_energy_in_ev_from_velocity_in_km_per_second(
        #       particle_velocity_sw_frame)

        # pitch_angle_bin_edges = [0, 9, 18, 27,  36, ..., 171, 180]
        #   flux_by_pa = rebin_by_pitch_angle(flux_data, pitch_angle, energy_in_sw_frame, pitch_angle_bin_edges, energy_bin_centers)
        #   psd_by_pa = rebin_by_pitch_angle(psd_data, pitch_angle, energy_in_sw_frame, pitch_angle_bin_edges, energy_bin_centers)

        #   flux_by_pa_and_gyrophase = rebin_by_pitch_angle(flux_data, pitch_angle, gyrophase, energy_in_sw_frame)
        #   psd_by_pa_gyrophase = rebin_by_pitch_angle(psd_data, pitch_angle, gyrophase, energy_in_sw_frame)

        pass
