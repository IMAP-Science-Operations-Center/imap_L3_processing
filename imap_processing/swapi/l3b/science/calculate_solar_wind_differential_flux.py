from numpy import ndarray

from imap_processing.swapi.l3b.science.geometric_factor_calibration_table import GeometricFactorCalibrationTable


def calculate_combined_solar_wind_diffential_flux(energies: ndarray, average_count_rates: ndarray,
                                                  efficiency: float,
                                                  geometric_factor_table: GeometricFactorCalibrationTable):
    geometric_factor = geometric_factor_table.lookup_geometric_factor(energies)
    denominator = energies * geometric_factor * efficiency
    return average_count_rates / denominator
