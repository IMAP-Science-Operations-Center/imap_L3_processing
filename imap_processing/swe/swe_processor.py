from imap_processing.processor import Processor


class SweProcessor(Processor):
    def process(self):
        # rebinned_mag = mag.rebin_to(epoch, epoch_delta)

        # iterate minute chucks of SweL3Data
        #   flux = calulate average flux
        #   spacecraft_potential, halo_core = find_breakpoints(energies ,flux)
        #
        #   subtract spacecraft_potential from energies
        #   compute_velocity_in_dsp_frame_km_s
        #   compute_velocity_in_sw_frame
        #   compute_energy_in_ev_from_velocity_in_km_per_second
        #   calculate_pitch_angle
        #   calculate_gyrophase
        pass
