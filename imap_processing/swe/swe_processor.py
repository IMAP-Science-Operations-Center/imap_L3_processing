from imap_processing.processor import Processor


class SweProcessor(Processor):
    def process(self):
        # rebinned_mag = mag.rebin_to(epoch, epoch_delta)

        # iterate minute chucks of SweL3Data
        #   flux = calulate average flux
        #   spacecraft_potential, halo_core = find_breakpoints(energies ,flux)
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
