from imap_processing.hit.l3.hit_l3_sectored_dependencies import HITL3SectoredDependencies
from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, calculate_sector_areas, rebin_by_pitch_angle_and_gyrophase
from imap_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase
from imap_processing.processor import Processor


class HitProcessor(Processor):
    def process(self):
        number_of_pitch_angle_bins = 8
        number_of_gyrophase_bins = 15

        dependencies = HITL3SectoredDependencies.fetch_dependencies(self.dependencies)
        mag_data = dependencies.mag_l1d_data
        hit_data = dependencies.data
        species_list = [hit_data.CNO, hit_data.helium4, hit_data.hydrogen, hit_data.iron, hit_data.NeMgSi]

        # sector directions are constant
        dec, dec_delta, inc, inc_delta = get_hit_bin_polar_coordinates()
        sector_unit_vectors = get_sector_unit_vectors(dec, inc)
        particle_unit_vectors = -sector_unit_vectors

        sector_areas = calculate_sector_areas(dec, dec_delta, inc_delta)

        averaged_mag_data = mag_data.rebin_to(hit_data.epoch, hit_data.epoch_delta)
        for time_index, average_mag_vector in enumerate(averaged_mag_data):
            mag_unit_vector = calculate_unit_vector(average_mag_vector)
            pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vector)
            gyrophases = calculate_gyrophase(particle_unit_vectors, mag_unit_vector)

            for species_hit_data in species_list:
                for energy in species_hit_data[time_index]:
                    rebinned_data = rebin_by_pitch_angle_and_gyrophase(energy, pitch_angles, gyrophases, sector_areas,
                                                                       number_of_pitch_angle_bins,
                                                                       number_of_gyrophase_bins)
