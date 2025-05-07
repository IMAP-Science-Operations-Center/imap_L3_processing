from collections import namedtuple

import numpy as np
from imap_data_access import upload
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.models import CodiceHiL3PitchAngleDataProduct
from imap_l3_processing.codice.l3.hi.models import CodiceL3HiDirectEvents
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import \
    hit_rebin_by_pitch_angle_and_gyrophase, get_sector_unit_vectors
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceHiProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self):
        if self.input_metadata.data_level == "l3a":
            dependencies = CodiceHiL3aDirectEventsDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_direct_event(dependencies)
        elif self.input_metadata.data_level == "l3b":
            dependencies = CodicePitchAngleDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3b(dependencies)
        else:
            raise NotImplementedError(f"Unknown data level for CoDICE: {self.input_metadata.data_level}")

        saved_cdf = save_data(data_product)
        upload(saved_cdf)

    def process_l3a_direct_event(self, dependencies: CodiceHiL3aDirectEventsDependencies) -> CodiceL3HiDirectEvents:
        tof_lookup = dependencies.tof_lookup
        l2_data = dependencies.codice_l2_hi_data

        (data_quality,
         num_of_events) = [np.full((len(l2_data.epochs), len(l2_data.priority_events)), np.nan) for _ in range(2)]

        (erge,
         multi_flag,
         ssd_energy,
         ssd_id,
         spin_angle,
         spin_number,
         tof,
         type,
         energy_per_nuc,
         estimated_mass) = [np.full((len(l2_data.epochs), len(l2_data.priority_events), len(num_of_events)), np.nan)
                            for _ in range(10)]

        for index, priority_event in enumerate(l2_data.priority_events):
            event_tof = priority_event.time_of_flight
            event_energy_per_nuc = np.array([tof_lookup[t].energy for t in event_tof.flat]).reshape(
                (event_tof.shape))
            event_estimated_mass = (priority_event.ssd_energy / event_energy_per_nuc)

            erge[:, index, :] = priority_event.energy_range
            multi_flag[:, index, :] = priority_event.multi_flag
            ssd_energy[:, index, :] = priority_event.ssd_energy
            ssd_id[:, index, :] = priority_event.ssd_id
            spin_angle[:, index, :] = priority_event.spin_angle
            spin_number[:, index, :] = priority_event.spin_number
            tof[:, index, :] = priority_event.time_of_flight
            type[:, index, :] = priority_event.type
            energy_per_nuc[:, index, :] = event_energy_per_nuc
            estimated_mass[:, index, :] = event_estimated_mass

            data_quality[:, index] = priority_event.data_quality
            num_of_events[:, index] = priority_event.number_of_events

        return CodiceL3HiDirectEvents(
            input_metadata=self.input_metadata,
            epoch=l2_data.epochs,
            data_quality=data_quality,
            erge=erge,
            multi_flag=multi_flag,
            num_of_events=num_of_events,
            ssd_energy=ssd_energy,
            ssd_id=ssd_id,
            spin_angle=spin_angle,
            spin_number=spin_number,
            tof=tof,
            type=type,
            energy_per_nuc=energy_per_nuc,
            estimated_mass=estimated_mass
        )

    def process_l3b(self, dependencies: CodicePitchAngleDependencies) -> CodiceHiL3PitchAngleDataProduct:
        mag_data = dependencies.mag_l1d_data
        sectored_intensities = dependencies.codice_sectored_intensities_data
        epochs = dependencies.codice_sectored_intensities_data.epoch
        energy_bins = dependencies.codice_sectored_intensities_data.energy
        rebinned_mag_data = mag_data.rebin_to(sectored_intensities.epoch, sectored_intensities.epoch_delta)

        mag_unit_vectors = calculate_unit_vector(rebinned_mag_data)
        sector_unit = get_sector_unit_vectors(sectored_intensities.spin_sector, sectored_intensities.ssd_id)
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
                    number_of_pitch_angle_bins=pitch_angles.shape[-1],
                    number_of_gyrophase_bins=gyrophase.shape[-1])

                species_intensity.intensity_by_pa[time_index] = rebinned_intensities_by_pa
                species_intensity.intensity_by_pa_and_gyro[time_index] = rebinned_intensities_by_pa_and_gyro

        return CodiceHiL3PitchAngleDataProduct(
            input_metadata=self.input_metadata,
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
