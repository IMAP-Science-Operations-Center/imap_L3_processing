from imap_l3_processing.swapi.l3a.models import SwapiL2Data


def extract_coarse_sweep(input_data: SwapiL2Data) -> SwapiL2Data:
    coarse_sweep_energies = input_data.energy[1:63]
    coarse_sweep_coincidence_count_rates = input_data.coincidence_count_rate[:, 1:63]
    coarse_sweep_spin_angles = input_data.spin_angles[:, 1:63]
    coarse_sweep_coincidence_count_rate_uncertainties = input_data.coincidence_count_rate_uncertainty[:, 1:63]

    return SwapiL2Data(input_data.epoch, coarse_sweep_energies, coarse_sweep_coincidence_count_rates,
                       coarse_sweep_spin_angles, coarse_sweep_coincidence_count_rate_uncertainties)
