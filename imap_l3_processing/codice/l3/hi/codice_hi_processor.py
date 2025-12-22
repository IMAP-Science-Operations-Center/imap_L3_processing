from collections import namedtuple
from pathlib import Path

import numpy as np
from imap_data_access.processing_input import ProcessingInputCollection

from imap_l3_processing.codice.l3.hi.direct_event.codice_hi_l3a_direct_events_dependencies import \
    CodiceHiL3aDirectEventsDependencies
from imap_l3_processing.codice.l3.hi.models import CodiceHiL3PitchAngleDataProduct
from imap_l3_processing.codice.l3.hi.models import CodiceL3HiDirectEvents
from imap_l3_processing.codice.l3.hi.pitch_angle.codice_pitch_angle_dependencies import CodicePitchAngleDependencies
from imap_l3_processing.codice.l3.lo.constants import CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM
from imap_l3_processing.hit.l3.sectored_products.science.sectored_products_algorithms import \
    get_sector_unit_vectors
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.pitch_angles import calculate_unit_vector, calculate_pitch_angle, calculate_gyrophase, \
    rebin_by_pitch_angle_and_gyrophase
from imap_l3_processing.processor import Processor
from imap_l3_processing.utils import save_data


class CodiceHiProcessor(Processor):
    def __init__(self, dependencies: ProcessingInputCollection, input_metadata: InputMetadata):
        super().__init__(dependencies, input_metadata)

    def process(self) -> list[Path]:
        if self.input_metadata.data_level == "l3a":
            dependencies = CodiceHiL3aDirectEventsDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3a_direct_event(dependencies)
        elif self.input_metadata.data_level == "l3b":
            dependencies = CodicePitchAngleDependencies.fetch_dependencies(self.dependencies)
            data_product = self.process_l3b(dependencies)
        else:
            raise NotImplementedError(f"Unknown data level for CoDICE: {self.input_metadata.data_level}")

        data_product.parent_file_names = self.get_parent_file_names()
        return [save_data(data_product)]

    def process_l3a_direct_event(self, dependencies: CodiceHiL3aDirectEventsDependencies) -> CodiceL3HiDirectEvents:
        l2_data = dependencies.codice_l2_hi_data

        estimated_mass = np.full_like(l2_data.energy_per_nuc, fill_value=np.nan)
        for i in range(l2_data.energy_per_nuc.shape[0]):
            for j in range(l2_data.energy_per_nuc.shape[1]):
                npts = l2_data.number_of_events[i, j]
                estimated_mass[i, j, :npts] = l2_data.ssd_energy[i, j, :npts] / l2_data.energy_per_nuc[i, j, :npts]
        return CodiceL3HiDirectEvents(
            input_metadata=self.input_metadata,
            epoch=l2_data.epoch,
            epoch_delta=l2_data.epoch_delta_plus,
            data_quality=l2_data.data_quality,
            multi_flag=l2_data.multi_flag,
            num_events=l2_data.number_of_events,
            ssd_energy=l2_data.ssd_energy,
            ssd_id=l2_data.ssd_id,
            spin_angle=l2_data.spin_angle,
            spin_number=l2_data.spin_number,
            tof=l2_data.time_of_flight,
            type=l2_data.type,
            energy_per_nuc=l2_data.energy_per_nuc,
            estimated_mass=estimated_mass
        )

    def process_l3b(self, dependencies: CodicePitchAngleDependencies) -> CodiceHiL3PitchAngleDataProduct:
        mag_data = dependencies.mag_l1d_data
        sectored_intensities = dependencies.codice_sectored_intensities_data
        epochs = dependencies.codice_sectored_intensities_data.epoch
        h_energy_bins = dependencies.codice_sectored_intensities_data.energy_h.shape[0]
        he3he4_energy_bins = dependencies.codice_sectored_intensities_data.energy_he3he4.shape[0]
        cno_energy_bins = dependencies.codice_sectored_intensities_data.energy_cno.shape[0]
        fe_energy_bins = dependencies.codice_sectored_intensities_data.energy_fe.shape[0]
        rebinned_mag_data = mag_data.rebin_to(sectored_intensities.epoch, sectored_intensities.epoch_delta_plus)

        spin_angles_in_dps = (sectored_intensities.spin_angles + CODICE_SPIN_ANGLE_OFFSET_FROM_MAG_BOOM) % 360

        mag_unit_vectors = calculate_unit_vector(rebinned_mag_data)
        sector_unit = get_sector_unit_vectors(sectored_intensities.elevation_angle, spin_angles_in_dps)
        sector_unit_vectors = calculate_unit_vector(sector_unit)
        particle_unit_vectors = -1 * sector_unit_vectors

        num_pitch_angle_bins = 6
        num_gyrophase_bins = 12
        pitch_angle_delta_value = (180 / num_pitch_angle_bins) / 2
        gyrophase_delta_value = (360 / num_gyrophase_bins) / 2
        pitch_angle_bins = np.linspace(0, 180, num_pitch_angle_bins, endpoint=False) + pitch_angle_delta_value
        gyrophase_bins = np.linspace(0, 360, num_gyrophase_bins, endpoint=False) + gyrophase_delta_value

        h_pa_shape = (len(epochs), h_energy_bins, num_pitch_angle_bins)
        he3he4_pa_shape = (len(epochs), he3he4_energy_bins, num_pitch_angle_bins)
        cno_pa_shape = (len(epochs), cno_energy_bins, num_pitch_angle_bins)
        fe_pa_shape = (len(epochs), fe_energy_bins, num_pitch_angle_bins)
        h_gyro_shape = (len(epochs), h_energy_bins, num_pitch_angle_bins, num_gyrophase_bins)
        he3he4_gyro_shape = (len(epochs), he3he4_energy_bins, num_pitch_angle_bins, num_gyrophase_bins)
        cno_gyro_shape = (len(epochs), cno_energy_bins, num_pitch_angle_bins, num_gyrophase_bins)
        fe_gyro_shape = (len(epochs), fe_energy_bins, num_pitch_angle_bins, num_gyrophase_bins)

        SpeciesIntensity = namedtuple("SpeciesIntensity",
                                      ["l2_intensity", "intensity_by_pa", "intensity_by_pa_and_gyro"])

        # @formatter:off
        species_intensities: dict[str, SpeciesIntensity] = {
            "h": SpeciesIntensity(sectored_intensities.h_intensities, *_create_pa_and_gyro_nan_arrays(h_pa_shape, h_gyro_shape)),
            "he4": SpeciesIntensity(sectored_intensities.he3he4_intensities, *_create_pa_and_gyro_nan_arrays(he3he4_pa_shape, he3he4_gyro_shape)),
            "o": SpeciesIntensity(sectored_intensities.cno_intensities, *_create_pa_and_gyro_nan_arrays(cno_pa_shape, cno_gyro_shape)),
            "fe": SpeciesIntensity(sectored_intensities.fe_intensities,*_create_pa_and_gyro_nan_arrays(fe_pa_shape, fe_gyro_shape))}
        # @formatter:on

        for time_index in range(len(epochs)):
            pitch_angles = calculate_pitch_angle(particle_unit_vectors, mag_unit_vectors[time_index])
            gyrophases = calculate_gyrophase(particle_unit_vectors, mag_unit_vectors[time_index])

            for species, species_intensity in species_intensities.items():
                intensity = species_intensity.l2_intensity[time_index]

                intensities_delta_plus = np.full_like(intensity, 0)
                intensities_delta_minus = np.full_like(intensity, 0)

                rebinned_intensities_by_pa_and_gyro, _, _, rebinned_intensities_by_pa, _, _ = rebin_by_pitch_angle_and_gyrophase(
                    intensity_data=intensity, intensity_delta_plus=intensities_delta_plus,
                    intensity_delta_minus=intensities_delta_minus, pitch_angles=pitch_angles, gyrophases=gyrophases,
                    number_of_pitch_angle_bins=num_pitch_angle_bins, number_of_gyrophase_bins=num_gyrophase_bins)

                species_intensity.intensity_by_pa[time_index] = rebinned_intensities_by_pa
                species_intensity.intensity_by_pa_and_gyro[time_index] = rebinned_intensities_by_pa_and_gyro

        return CodiceHiL3PitchAngleDataProduct(
            input_metadata=self.input_metadata,
            epoch=epochs,
            epoch_delta=sectored_intensities.epoch_delta_plus,
            energy_h=dependencies.codice_sectored_intensities_data.energy_h,
            energy_h_plus=dependencies.codice_sectored_intensities_data.energy_h_plus,
            energy_h_minus=dependencies.codice_sectored_intensities_data.energy_h_minus,
            energy_cno=dependencies.codice_sectored_intensities_data.energy_cno,
            energy_cno_plus=dependencies.codice_sectored_intensities_data.energy_cno_plus,
            energy_cno_minus=dependencies.codice_sectored_intensities_data.energy_cno_minus,
            energy_fe=dependencies.codice_sectored_intensities_data.energy_fe,
            energy_fe_plus=dependencies.codice_sectored_intensities_data.energy_fe_plus,
            energy_fe_minus=dependencies.codice_sectored_intensities_data.energy_fe_minus,
            energy_he3he4=dependencies.codice_sectored_intensities_data.energy_he3he4,
            energy_he3he4_plus=dependencies.codice_sectored_intensities_data.energy_he3he4_plus,
            energy_he3he4_minus=dependencies.codice_sectored_intensities_data.energy_he3he4_minus,
            pitch_angle=pitch_angle_bins,
            pitch_angle_delta=np.repeat(pitch_angle_delta_value, num_pitch_angle_bins),
            gyrophase=gyrophase_bins,
            gyrophase_delta=np.repeat(gyrophase_delta_value, num_gyrophase_bins),
            h_intensity_by_pitch_angle=species_intensities['h'].intensity_by_pa,
            h_intensity_by_pitch_angle_and_gyrophase=species_intensities['h'].intensity_by_pa_and_gyro,
            he3he4_intensity_by_pitch_angle=species_intensities['he4'].intensity_by_pa,
            he3he4_intensity_by_pitch_angle_and_gyrophase=species_intensities['he4'].intensity_by_pa_and_gyro,
            cno_intensity_by_pitch_angle=species_intensities['o'].intensity_by_pa,
            cno_intensity_by_pitch_angle_and_gyrophase=species_intensities['o'].intensity_by_pa_and_gyro,
            fe_intensity_by_pitch_angle=species_intensities['fe'].intensity_by_pa,
            fe_intensity_by_pitch_angle_and_gyrophase=species_intensities['fe'].intensity_by_pa_and_gyro
        )


def _create_pa_and_gyro_nan_arrays(pitch_angle_shape, pitch_angle_and_gyrophase_shape) -> np.array:
    return np.full(pitch_angle_shape, fill_value=np.nan), np.full(pitch_angle_and_gyrophase_shape, fill_value=np.nan)
