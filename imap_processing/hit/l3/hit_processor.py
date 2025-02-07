import numpy as np

from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates, calculate_pitch_angle, calculate_gyrophase, rebin_by_pitch_angle_and_gyrophase, \
    calculate_unit_vector, calculate_sector_areas
from imap_processing.processor import Processor


class HitProcessor(Processor):
    def process(self):
        for species in species_list:
            species_hit_data = hit_data[species]
            for mag_chunk, hit_l2_chunk in get_ten_minute_chunk(mag_data, species_hit_data):
                average_mag_vector = np.average(mag_chunk[:, :3], axis=0)
                mag_unit_vector = calculate_unit_vector(average_mag_vector)

                dec, dec_delta, inc, inc_delta = get_hit_bin_polar_coordinates()
                sector_unit_vectors = get_sector_unit_vectors(dec, inc)
                sector_areas = calculate_sector_areas(dec, dec_delta, inc_delta)
                particle_unit_vectors = -sector_unit_vectors
                pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vector)

                gyrophases = calculate_gyrophase(particle_unit_vectors, mag_unit_vector)

                number_of_pitch_angle_bins = 8
                number_of_gyrophase_bins = 15
                rebinned_data = rebin_by_pitch_angle_and_gyrophase(hit_l2_chunk, pitch_angles, gyrophases, sector_areas,
                                                                   number_of_pitch_angle_bins, number_of_gyrophase_bins)
