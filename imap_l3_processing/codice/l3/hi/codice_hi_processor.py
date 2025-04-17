from collections import namedtuple

import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3_dependencies import CodiceHiL3Dependencies
from imap_l3_processing.codice.l3.hi.models import CodiceHiL3PitchAngleDataProduct
from imap_l3_processing.codice.l3.hi.models import CodiceL3HiDirectEvents, CodiceL3HiDirectEventsBuilder
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import \
    hit_rebin_by_pitch_angle_and_gyrophase, get_sector_unit_vectors_codice
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceHiProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            dependencies = CodiceHiL3Dependencies.fetch_dependencies(self.dependencies)
            processed_codice_direct_events = self.process_l3a(dependencies)
            saved_cdf = save_data(processed_codice_direct_events)
            upload(saved_cdf)

    def process_l3a(self, dependencies: CodiceHiL3Dependencies) -> CodiceL3HiDirectEvents:
        tof_lookup = dependencies.tof_lookup
        l2_data = dependencies.codice_l2_hi_data

        estimated_mass_with_bounds = []
        energy_per_nucleons_per_priority_event = []
        for index, priority_event in enumerate(l2_data.priority_events):
            tof = priority_event.time_of_flight
            energy_per_nuc_with_bounds = np.array([e for t in tof.flat for e in tof_lookup[t]]).reshape((*tof.shape, 3))
            estimated_mass_with_bounds.append(priority_event.ssd_energy[:, :, np.newaxis] / energy_per_nuc_with_bounds)
            energy_per_nucleons_per_priority_event.append(energy_per_nuc_with_bounds)

        # @formatter:off
        return (CodiceL3HiDirectEventsBuilder(l2_data)
                            .updated_priority_event_0(energy_per_nucleons_per_priority_event[0], estimated_mass_with_bounds[0])
                            .updated_priority_event_1(energy_per_nucleons_per_priority_event[1], estimated_mass_with_bounds[1])
                            .updated_priority_event_2(energy_per_nucleons_per_priority_event[2], estimated_mass_with_bounds[2])
                            .updated_priority_event_3(energy_per_nucleons_per_priority_event[3], estimated_mass_with_bounds[3])
                            .updated_priority_event_4(energy_per_nucleons_per_priority_event[4], estimated_mass_with_bounds[4])
                            .updated_priority_event_5(energy_per_nucleons_per_priority_event[5], estimated_mass_with_bounds[5])
                            .convert())
        # @formatter:on

    def process_l3b(self, dependencies: CodicePitchAngleDependencies) -> CodiceHiL3PitchAngleDataProduct:
        mag_data = dependencies.mag_l1d_data
        sectored_intensities = dependencies.codice_sectored_intensities_data
        epochs = dependencies.codice_sectored_intensities_data.epoch
        energy_bins = dependencies.codice_sectored_intensities_data.energy
        rebinned_mag_data = mag_data.rebin_to(sectored_intensities.epoch, sectored_intensities.epoch_delta)

        mag_unit_vectors = calculate_unit_vector(rebinned_mag_data)
        sector_unit = get_sector_unit_vectors_codice(sectored_intensities.spin_sector, sectored_intensities.ssd_id)
        sector_unit_vectors = calculate_unit_vector(sector_unit)
        particle_unit_vectors = -1 * sector_unit_vectors
        pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vectors)
        pitch_angle_delta_value = 15
        gyrophase_delta_value = 30
        pitch_angle_delta = np.repeat(pitch_angle_delta_value, len(pitch_angles))
        gyrophase_delta = np.repeat(gyrophase_delta_value, len(pitch_angles))
        gyrophase = calculate_gyrophase(particle_unit_vectors, mag_unit_vectors)
        num_pitch_angles = len(pitch_angles)
        num_gyrophases = len(gyrophase)
        pa_shape = (len(epochs), len(energy_bins), num_pitch_angles)
        gyro_shape = (len(epochs), len(energy_bins), num_pitch_angles, num_gyrophases)

        SpeciesIntensity = namedtuple("SpeciesIntensity",
                                      ["l2_intensity", "intensity_by_pa", "intensity_by_pa_and_gyro"])

        # @formatter:off
        species_intensities: dict[str, SpeciesIntensity] = {
            "h": SpeciesIntensity(sectored_intensities.h_intensities, *_create_pa_and_gyro_nan_arrays(pa_shape, gyro_shape)),
            "he4": SpeciesIntensity(sectored_intensities.he4_intensities, *_create_pa_and_gyro_nan_arrays(pa_shape, gyro_shape)),
            "o": SpeciesIntensity(sectored_intensities.o_intensities, *_create_pa_and_gyro_nan_arrays(pa_shape, gyro_shape)),
            "fe": SpeciesIntensity(sectored_intensities.fe_intensities,*_create_pa_and_gyro_nan_arrays(pa_shape, gyro_shape))}
        # @formatter:on

        for species, species_intensity in species_intensities.items():
            for time_index in range(len(epochs)):
                intensity = species_intensity.l2_intensity[time_index]

                h_intensities_delta_plus = np.full_like(intensity, 0)
                h_intensities_delta_minus = np.full_like(intensity, 0)

                rebinned_intensities_by_pa_and_gyro, _, _, rebinned_intensities_by_pa, _, _ = hit_rebin_by_pitch_angle_and_gyrophase(
                    intensity_data=intensity,
                    intensity_delta_plus=h_intensities_delta_plus,
                    intensity_delta_minus=h_intensities_delta_minus,
                    gyrophases=gyrophase,
                    pitch_angles=pitch_angles,
                    number_of_pitch_angle_bins=2,
                    number_of_gyrophase_bins=2)

                species_intensity.intensity_by_pa[time_index] = rebinned_intensities_by_pa
                species_intensity.intensity_by_pa_and_gyro[time_index] = rebinned_intensities_by_pa_and_gyro

        return CodiceHiL3PitchAngleDataProduct(
            input_metadata=None,
            epoch=epochs,
            epoch_delta=sectored_intensities.epoch_delta,
            energy=energy_bins,
            energy_delta_plus=sectored_intensities.energy_delta_plus,
            energy_delta_minus=sectored_intensities.energy_delta_minus,
            pitch_angle=pitch_angles,
            pitch_angle_delta=pitch_angle_delta,
            gyrophase=gyrophase,
            gyrophase_delta=gyrophase_delta,
            h_intensity_by_pitch_angle=species_intensities['h'].intensity_by_pa,
            h_intensity_by_pitch_angle_and_gyrophase=species_intensities['h'].intensity_by_pa_and_gyro,
            he4_intensity_by_pitch_angle=species_intensities['he4'].intensity_by_pa,
            he4_intensity_by_pitch_angle_and_gyrophase=species_intensities['he4'].intensity_by_pa_and_gyro,
            o_intensity_by_pitch_angle=species_intensities['o'].intensity_by_pa,
            o_intensity_by_pitch_angle_and_gyrophase=species_intensities['o'].intensity_by_pa_and_gyro,
            fe_intensity_by_pitch_angle=species_intensities['fe'].intensity_by_pa,
            fe_intensity_by_pitch_angle_and_gyrophase=species_intensities['fe'].intensity_by_pa_and_gyro
        )


def _create_pa_and_gyro_nan_arrays(pitch_angle_shape, pitch_angle_and_gyrophase_shape) -> np.array:
    return np.full(pitch_angle_shape, fill_value=np.nan), np.full(pitch_angle_and_gyrophase_shape, fill_value=np.nan)
