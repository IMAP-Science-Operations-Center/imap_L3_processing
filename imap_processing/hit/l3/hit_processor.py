import imap_data_access
import numpy as np
from spacepy.pycdf import CDF
from spiceypy import spiceypy

from imap_processing import utils
from imap_processing.hit.l3.models import HitL2Data
from imap_processing.hit.l3.sectored_products.science.sectored_products_algorithms import get_sector_unit_vectors, \
    get_hit_bin_polar_coordinates
from imap_processing.hit.l3.utils import read_l2_hit_data, calculate_unit_vector, calculate_pitch_angle
from imap_processing.processor import Processor
from imap_processing.utils import format_time, read_l1d_mag_data


class HitProcessor(Processor):
    def process(self):
        for species in species_list:
            species_hit_data = hit_data[species]
            for mag_chunk, hit_l2_chunk in get_ten_minute_chunk(mag_data, species_hit_data):
                average_mag_vector = np.average(mag_chunk[:, :3], axis=0)
                mag_unit_vector = calculate_unit_vector(average_mag_vector)

                sector_unit_vectors = get_sector_unit_vectors()
                particle_unit_vectors = -sector_unit_vectors
                pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vector)

                dps_unit_x_vectors = [1, 0, 0]
                gyrophases = calculate_gyrophase(mag_unit_vector, particle_unit_vectors, dps_unit_x_vectors)

                number_of_pitch_angle_bins = 8
                number_of_gyrophase_bins = 15
                rebinned_data = rebin_by_pitch_angle_and_gyrophase(hit_l2_chunk, pitch_angles, gyrophases,
                                                                   number_of_pitch_angle_bins, number_of_gyrophase_bins)
