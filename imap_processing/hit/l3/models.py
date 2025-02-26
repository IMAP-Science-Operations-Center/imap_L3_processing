from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from bitstring import BitStream

from imap_processing.hit.l3.pha.science.calculate_pha import EventOutput
from imap_processing.models import DataProduct, DataProductVariable

EPOCH_VAR_NAME = "EPOCH"
CHARGE_VAR_NAME = "CHARGE"
ENERGY_VAR_NAME = "ENERGY"
PARTICLE_ID_VAR_NAME = "PARTICLE_ID"
PRIORITY_BUFFER_ID_VAR_NAME = "PRIORITY_BUFFER_ID"
LATENCY_VAR_NAME = "LATENCY"
STIM_FLAG_VAR_NAME = "STIM_FLAG"
LONG_EVENT_FLAG_VAR_NAME = "LONG_EVENT_FLAG"
HAZ_FLAG_VAR_NAME = "HAZ_FLAG"
A_B_SIDE_VAR_NAME = "A_B_SIDE"
HAS_UNREAD_FLAG_VAR_NAME = "HAS_UNREAD_FLAG"
CULLING_FLAG_VAR_NAME = "CULLING_FLAG"
PHA_VALUE_VAR_NAME = "PHA_VALUE"
ENERGY_AT_DETECTOR_VAR_NAME = "ENERGY_AT_DETECTOR"
DETECTOR_ADDRESS_VAR_NAME = "DETECTOR_ADDRESS"
GAIN_FLAG_VAR_NAME = "GAIN_FLAG"
LAST_PHA_VAR_NAME = "LAST_PHA"
DETECTOR_FLAG_VAR_NAME = "DETECTOR_FLAG"
DEINDEX_VAR_NAME = "DEINDEX"
EPINDEX_VAR_NAME = "EPINDEX"
STIM_GAIN_VAR_NAME = "STIM_GAIN"
A_L_STIM_VAR_NAME = "A_L_STIM"
STIM_STEP_VAR_NAME = "STIM_STEP"
DAC_VALUE_VAR_NAME = "DAC_VALUE"


@dataclass
class HitL2Data:
    epoch: np.ndarray[datetime]
    epoch_delta: np.ndarray[timedelta]
    hydrogen: np.ndarray[float]
    helium4: np.ndarray[float]
    CNO: np.ndarray[float]
    NeMgSi: np.ndarray[float]
    iron: np.ndarray[float]

    DELTA_MINUS_CNO: np.ndarray[float]
    DELTA_MINUS_HELIUM4: np.ndarray[float]
    DELTA_MINUS_HYDROGEN: np.ndarray[float]
    DELTA_MINUS_IRON: np.ndarray[float]
    DELTA_MINUS_NEMGSI: np.ndarray[float]
    DELTA_PLUS_CNO: np.ndarray[float]
    DELTA_PLUS_HELIUM4: np.ndarray[float]
    DELTA_PLUS_HYDROGEN: np.ndarray[float]
    DELTA_PLUS_IRON: np.ndarray[float]
    DELTA_PLUS_NEMGSI: np.ndarray[float]
    cno_energy_high: np.ndarray[float]
    cno_energy_idx: np.ndarray[int]
    cno_energy_low: np.ndarray[float]
    fe_energy_high: np.ndarray[float]
    fe_energy_idx: np.ndarray[int]
    fe_energy_low: np.ndarray[float]
    h_energy_high: np.ndarray[float]
    h_energy_idx: np.ndarray[int]
    h_energy_low: np.ndarray[float]
    he4_energy_high: np.ndarray[float]
    he4_energy_idx: np.ndarray[int]
    he4_energy_low: np.ndarray[float]
    nemgsi_energy_high: np.ndarray[float]
    nemgsi_energy_idx: np.ndarray[int]
    nemgsi_energy_low: np.ndarray[float]


@dataclass
class HitDirectEventDataProduct(DataProduct):
    event_outputs: list[EventOutput]

    def to_data_product_variables(self) -> list[DataProductVariable]:
        charges = []
        energies = []
        particle_ids = []
        priority_buffer_numbers = []
        time_tags = []
        stim_tags = []
        long_event_flags = []
        haz_tags = []
        a_b_sides = []
        has_unread_adcs = []
        culling_flags = []
        pha_values = []
        energies_per_detector = []
        detector_addresses = []
        gain_flags = []
        last_phas = []
        detector_flags = []
        de_index = []
        ep_index = []
        stim_gains = []
        a_l_stims = []
        stim_steps = []
        dac_values = []

        for event_output in self.event_outputs:
            charges.append(event_output.charge)
            energies.append(event_output.total_energy)
            particle_ids.append(event_output.original_event.particle_id)
            priority_buffer_numbers.append(event_output.original_event.priority_buffer_num)
            time_tags.append(event_output.original_event.time_tag)
            stim_tags.append(event_output.original_event.stim_tag)
            long_event_flags.append(event_output.original_event.long_event_flag)
            haz_tags.append(event_output.original_event.haz_tag)
            a_b_sides.append(event_output.original_event.a_b_side_flag)
            has_unread_adcs.append(event_output.original_event.has_unread_adcs)
            culling_flags.append(event_output.original_event.culling_flag)
            pha_values.append([pha_word.adc_value for pha_word in event_output.original_event.pha_words])
            energies_per_detector.append(event_output.energies)
            detector_addresses.append([pha_word.detector.address for pha_word in event_output.original_event.pha_words])
            gain_flags.append([pha_word.is_low_gain for pha_word in event_output.original_event.pha_words])
            last_phas.append([pha_word.is_last_pha for pha_word in event_output.original_event.pha_words])
            self._process_extended_header(de_index, detector_flags, ep_index, event_output)
            self._process_stim_block(a_l_stims, event_output, stim_gains, stim_steps)
            self._process_extended_stim_header(dac_values, event_output)
        return [
            DataProductVariable(CHARGE_VAR_NAME, np.array(charges)),
            DataProductVariable(ENERGY_VAR_NAME, np.array(energies)),
            DataProductVariable(PARTICLE_ID_VAR_NAME, np.array(particle_ids)),
            DataProductVariable(PRIORITY_BUFFER_ID_VAR_NAME, np.array(priority_buffer_numbers)),
            DataProductVariable(LATENCY_VAR_NAME, np.array(time_tags)),
            DataProductVariable(STIM_FLAG_VAR_NAME, np.array(stim_tags)),
            DataProductVariable(LONG_EVENT_FLAG_VAR_NAME, np.array(long_event_flags)),
            DataProductVariable(HAZ_FLAG_VAR_NAME, np.array(haz_tags)),
            DataProductVariable(A_B_SIDE_VAR_NAME, np.array(a_b_sides)),
            DataProductVariable(HAS_UNREAD_FLAG_VAR_NAME, np.array(has_unread_adcs)),
            DataProductVariable(CULLING_FLAG_VAR_NAME, np.array(culling_flags)),
            DataProductVariable(PHA_VALUE_VAR_NAME, np.array(pha_values)),
            DataProductVariable(ENERGY_AT_DETECTOR_VAR_NAME, np.array(energies_per_detector)),
            DataProductVariable(DETECTOR_ADDRESS_VAR_NAME, np.array(detector_addresses)),
            DataProductVariable(GAIN_FLAG_VAR_NAME, np.array(gain_flags)),
            DataProductVariable(LAST_PHA_VAR_NAME, np.array(last_phas)),
            DataProductVariable(DETECTOR_FLAG_VAR_NAME, np.array(detector_flags)),
            DataProductVariable(DEINDEX_VAR_NAME, np.array(de_index)),
            DataProductVariable(EPINDEX_VAR_NAME, np.array(ep_index)),
            DataProductVariable(STIM_GAIN_VAR_NAME, np.array(stim_gains)),
            DataProductVariable(A_L_STIM_VAR_NAME, np.array(a_l_stims)),
            DataProductVariable(STIM_STEP_VAR_NAME, np.array(stim_steps)),
            DataProductVariable(DAC_VALUE_VAR_NAME, np.array(dac_values))
        ]

    def _process_stim_block(self, a_l_stims, event_output, stim_gains, stim_steps):
        if event_output.original_event.stim_block is None:
            stim_gains.append(None)
            a_l_stims.append(None)
            stim_steps.append(None)
        else:
            stim_gains.append(event_output.original_event.stim_block.stim_gain)
            a_l_stims.append(event_output.original_event.stim_block.a_l_stim)
            stim_steps.append(event_output.original_event.stim_block.stim_step)

    def _process_extended_header(self, de_index, detector_flags, ep_index, event_output):
        if event_output.original_event.extended_header is None:
            detector_flags.append(None)
            de_index.append(None)
            ep_index.append(None)
        else:
            detector_flags.append(event_output.original_event.extended_header.detector_flags)
            de_index.append(event_output.original_event.extended_header.delta_e_index)
            ep_index.append(event_output.original_event.extended_header.e_prime_index)

    def _process_extended_stim_header(self, dac_values, event_output):
        if event_output.original_event.extended_stim_header is None:
            dac_values.append(None)
        else:
            dac_values.append(event_output.original_event.extended_stim_header.dac_value)


@dataclass
class HitL1Data:
    epoch: np.ndarray[datetime]
    event_binary: list[BitStream]

    @classmethod
    def read_from_cdf(cls, cdf_file_path: Path):
        return cls(None, None)
