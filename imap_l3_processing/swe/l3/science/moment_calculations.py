import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union

import numpy as np
from spiceypy import spiceypy

from imap_l3_processing.constants import ELECTRON_MASS_KG, \
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN, ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR, METERS_PER_KILOMETER, \
    CENTIMETERS_PER_METER

# units of K / (m^2/s^2)
ZMK = ELECTRON_MASS_KG / (2 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)
NUMBER_OF_DETECTORS = 7


@dataclass
class Moments:
    alpha: float
    beta: float
    t_parallel: float
    t_perpendicular: float
    velocity_x: float
    velocity_y: float
    velocity_z: float
    density: float
    aoo: float
    ao: float

    @classmethod
    def construct_all_fill(cls):
        return cls(alpha=np.nan,
                   beta=np.nan,
                   t_parallel=np.nan,
                   t_perpendicular=np.nan,
                   velocity_x=np.nan,
                   velocity_y=np.nan,
                   velocity_z=np.nan,
                   density=np.nan,
                   aoo=np.nan,
                   ao=np.nan,
                   )


@dataclass
class MomentFitResults:
    moments: Moments
    chisq: float
    number_of_points: int


@dataclass
class HaloCorrectionParameters:
    spacecraft_potential: float
    core_halo_breakpoint: float


@dataclass
class CoreIntegrateOutputs:
    density: np.float64
    velocity: np.ndarray
    temperature: np.ndarray
    heat_flux: np.ndarray


@dataclass
class ScaleDensityOutput:
    density: np.float64
    velocity: np.ndarray
    temperature: np.ndarray
    cdelnv: np.ndarray
    cdelt: np.ndarray


def core_fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray,
                                         velocity_vectors: np.ndarray,
                                         phase_space_density: np.ndarray,
                                         weights: np.ndarray,
                                         energy_start: int,
                                         energy_end: int,
                                         density_history: Union[np.ndarray, list[float]]) -> Optional[MomentFitResults]:
    return _fit_moments_retrying_on_failure(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end,
        density_history,
        1.85
    )


def halo_fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray, velocity_vectors: np.ndarray,
                                         phase_space_density: np.ndarray,
                                         weights: np.ndarray,
                                         energy_start: int,
                                         energy_end: int,
                                         density_history: Union[np.ndarray, list[float]],
                                         spacecraft_potential: Union[np.float64, float],
                                         core_halo_breakpoint: Union[np.float64, float]) -> Optional[MomentFitResults]:
    return _fit_moments_retrying_on_failure(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end,
        density_history,
        1.35,
        halo_correction_parameters=HaloCorrectionParameters(spacecraft_potential,
                                                            core_halo_breakpoint)
    )


def _fit_moments_retrying_on_failure(corrected_energy_bins: np.ndarray,
                                     velocity_vectors: np.ndarray,
                                     phase_space_density: np.ndarray,
                                     weights: np.ndarray,
                                     energy_start: int,
                                     energy_end: int,
                                     density_history: Union[np.ndarray, list[float]],
                                     history_scalar: float,
                                     halo_correction_parameters: Optional[HaloCorrectionParameters] = None) -> Optional[
    MomentFitResults]:
    filtered_velocity_vectors, filtered_weights, filtered_yreg = filter_and_flatten_regress_parameters(
        corrected_energy_bins,
        velocity_vectors,
        phase_space_density,
        weights,
        energy_start,
        energy_end)

    fit_function, chi_squared = regress(filtered_velocity_vectors, filtered_weights, filtered_yreg)
    moment = calculate_fit_temperature_density_velocity(fit_function)
    average_density = np.average(density_history) * history_scalar

    if halo_correction_parameters is not None:
        moment.density = halotrunc(moment, halo_correction_parameters.core_halo_breakpoint,
                                   halo_correction_parameters.spacecraft_potential)

    if 0 < moment.density < average_density:
        return MomentFitResults(moments=moment, chisq=chi_squared, number_of_points=energy_end - energy_start)
    elif energy_end - energy_start < 4:
        return None
    else:
        return _fit_moments_retrying_on_failure(
            corrected_energy_bins,
            velocity_vectors,
            phase_space_density,
            weights,
            energy_start,
            energy_end - 1,
            density_history,
            history_scalar,
            halo_correction_parameters
        )


def regress(velocity_vectors: np.ndarray, weight: np.ndarray, yreg: np.ndarray) -> \
        np.ndarray:
    fit_function = np.zeros((len(velocity_vectors), 9))

    velocity_xs = velocity_vectors[:, 0]
    velocity_ys = velocity_vectors[:, 1]
    velocity_zs = velocity_vectors[:, 2]

    fit_function[:, 0] = -1 * ZMK * (velocity_xs ** 2)
    fit_function[:, 1] = -1 * ZMK * (velocity_ys ** 2)
    fit_function[:, 2] = -1 * ZMK * (velocity_zs ** 2)
    fit_function[:, 3] = -1 * ZMK * 2 * velocity_xs * velocity_ys
    fit_function[:, 4] = -1 * ZMK * 2 * velocity_xs * velocity_zs
    fit_function[:, 5] = -1 * ZMK * 2 * velocity_ys * velocity_zs
    fit_function[:, 6] = -1 * ZMK * 2 * velocity_xs
    fit_function[:, 7] = -1 * ZMK * 2 * velocity_ys
    fit_function[:, 8] = -1 * ZMK * 2 * velocity_zs

    r = np.zeros(9)
    a = np.zeros(10)
    sa = np.zeros(9)

    xmean = np.zeros(9)
    chisq = 0

    array = np.zeros((9, 9))

    wt = 1 / (weight * weight)
    sum = np.sum(wt)

    if sum == 0.0:
        return

    ymean = np.average(yreg, weights=wt)
    for i in range(9):
        xmean[i] = np.average(fit_function[:, i], weights=wt)

    wt /= sum

    yy = yreg - ymean

    for j in range(9):
        xx = fit_function[:, j] - xmean[j]
        r[j] = np.sum(wt * xx * yy)
        for k in range(j, 9):
            array[j][k] = np.sum(wt * xx * (fit_function[:, k] - xmean[k]))

    for j in range(9):
        for k in range(9):
            array[k][j] = array[j][k]

    # TODO the C code has an error case here, the following line can throw LinAlgError which may or may not be what the C code handles
    invarray = np.linalg.inv(array)

    ao = ymean

    yfit = np.zeros(len(fit_function))
    for j in range(9):
        a[j] = np.sum(r * invarray[j, :])
        ao -= a[j] * xmean[j]
        yfit += a[j] * fit_function[:, j]

    a[9] = ao
    for i in range(len(velocity_vectors)):
        yfit[i] += ao
        chisq += wt[i] * (yfit[i] - yreg[i]) * (yfit[i] - yreg[i])

    freen = len(velocity_vectors) - 9 - 1
    if freen > 0:
        chisq *= sum / freen
    elif freen == 0:
        chisq = 0.0

    for j in range(9):
        for k in range(9):
            invarray[j][k] /= sum
            sa[j] -= xmean[k] * invarray[j][k]

    sao = 1 / sum
    for j in range(9):
        for k in range(9):
            sao += xmean[j] * xmean[k] * invarray[j][k]

    return a, chisq


def calculate_fit_temperature_density_velocity(parameters: np.ndarray[float]):
    moments = Moments(alpha=0,
                      beta=0,
                      t_parallel=0,
                      t_perpendicular=0,
                      velocity_x=0,
                      velocity_y=0,
                      velocity_z=0,
                      density=0,
                      ao=0,
                      aoo=0
                      )

    moments.ao = parameters[9]
    if moments.ao == 0:
        return moments
    moments.alpha = np.atan2(parameters[5], parameters[4])
    moments.beta = np.atan2(parameters[3], parameters[4] * np.sin(moments.alpha))

    if moments.beta == 0.0:
        moments.t_perpendicular = 1.0 / parameters[0]
        moments.t_parallel = 1.0 / parameters[2]
    else:
        moments.t_perpendicular = 1.0 / (parameters[0] - parameters[3] * parameters[4] / parameters[5])
        moments.t_parallel = 1.0 / (parameters[0] + parameters[1] + parameters[2] - 2.0 / moments.t_perpendicular)

    c11 = parameters[1] * parameters[2] - parameters[5] * parameters[5]
    c12 = parameters[5] * parameters[4] - parameters[3] * parameters[2]
    c13 = parameters[3] * parameters[5] - parameters[1] * parameters[4]
    c22 = parameters[0] * parameters[2] - parameters[4] * parameters[4]
    c23 = parameters[4] * parameters[3] - parameters[0] * parameters[5]
    c33 = parameters[0] * parameters[1] - parameters[3] * parameters[3]

    d = parameters[0] * c11 + parameters[3] * c12 + parameters[4] * c13

    if d == 0:
        moments.velocity_x = 0
        moments.velocity_y = 0
        moments.velocity_z = 0
    else:
        moments.velocity_x = -1 * (parameters[6] * c11 + parameters[7] * c12 + parameters[8] * c13) / d
        moments.velocity_y = -1 * (parameters[6] * c12 + parameters[7] * c22 + parameters[8] * c23) / d
        moments.velocity_z = -1 * (parameters[6] * c13 + parameters[7] * c23 + parameters[8] * c33) / d

    fact = parameters[0] * moments.velocity_x ** 2 + parameters[1] * moments.velocity_y ** 2 + parameters[
        2] * moments.velocity_z ** 2

    moments.aoo = fact + parameters[3] * moments.velocity_x * moments.velocity_y + parameters[4] \
                  * moments.velocity_x * moments.velocity_z + \
                  parameters[5] * moments.velocity_y * moments.velocity_z

    moments.velocity_x *= 1e-5
    moments.velocity_y *= 1e-5
    moments.velocity_z *= 1e-5

    if (moments.t_parallel < 0) or (moments.aoo < 0) or (moments.aoo > 1e13):
        moments.density = 0.0
    else:
        moments.density = (moments.t_perpendicular * np.sqrt(moments.t_parallel) /
                           ((ZMK / math.pi) * np.sqrt(ZMK / math.pi))) * np.exp(
            moments.ao + ZMK * moments.aoo)
    return moments


LIMIT = np.array([32, 64, 128, 256, 1024, 2048, 3072, 5120, 9216, 17408, 33792])
SIGMA2 = np.array([0, 0.25, 1.25, 5.25, 21.25, 85.25, 341.25, 1365.25, 5461.25, 21845.25, 87381.25])
MINIMUM_WEIGHT = 0.8165
MAX_VARIANCE = 349525.25


def compute_maxwellian_weight_factors(count_rates: np.ndarray, acquisition_durations: np.ndarray) -> \
        np.ndarray[float]:
    correction = 1.0 - 1e-9 * LIMIT / acquisition_durations[:, :, np.newaxis]
    correction[correction < 0.1] = 0.1
    xlimits_per_measurement = LIMIT / correction

    counts = count_rates * acquisition_durations[:, :, np.newaxis]
    weights = np.empty_like(count_rates, dtype=np.float64)

    for (energy_i, spin_i, declination_i), corrected_count in np.ndenumerate(counts):
        if corrected_count <= 1.5:
            weights[energy_i, spin_i, declination_i] = MINIMUM_WEIGHT
        else:
            variance = MAX_VARIANCE
            xlimits = xlimits_per_measurement[energy_i, spin_i]
            for xlimit, sigma in zip(xlimits, SIGMA2):
                if corrected_count < xlimit:
                    variance = sigma
                    break
            weights[energy_i, spin_i, declination_i] = np.sqrt(variance + corrected_count) / corrected_count

    return weights


def filter_and_flatten_regress_parameters(corrected_energy_bins: np.ndarray,
                                          velocity_vectors: np.ndarray,
                                          phase_space_density: np.ndarray,
                                          weights: np.ndarray,
                                          core_breakpoint_index: int,
                                          core_halo_breakpoint_index: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = np.full_like(phase_space_density, fill_value=False, dtype=bool)
    valid_mask[core_breakpoint_index:core_halo_breakpoint_index] = True
    valid_mask[corrected_energy_bins <= 0] = False
    valid_mask[phase_space_density <= 0] = False

    filtered_phase_space_density = phase_space_density[valid_mask]
    yreg = np.zeros_like(filtered_phase_space_density)
    yreg[filtered_phase_space_density > 1e-35] = np.log(
        filtered_phase_space_density[filtered_phase_space_density > 1e-35])
    yreg[filtered_phase_space_density <= 1e-35] = -80.6

    return velocity_vectors[valid_mask], weights[valid_mask], yreg


def rotate_dps_vector_to_rtn(epoch: datetime, vector: np.ndarray) -> np.ndarray:
    et_time = spiceypy.datetime2et(epoch)
    rotation_matrix = spiceypy.pxform("IMAP_DPS", "IMAP_RTN", et_time)
    return rotation_matrix @ vector


def rotate_temperature(epoch: datetime, alpha: float, beta: float):
    sin_dec = np.sin(beta)
    x = sin_dec * np.cos(alpha)
    y = sin_dec * np.sin(alpha)
    z = np.cos(beta)

    rtn_temperature = rotate_dps_vector_to_rtn(epoch, np.array([x, y, z]))

    theta = np.asin(rtn_temperature[2])
    phi = np.atan2(rtn_temperature[1], rtn_temperature[0])

    return theta, phi


# this function computes 'dscale' from the heritage code
def compute_density_scale(core_electron_energy_range: float, speed: float, temperature: float) -> float:
    energy_multplier = 11600.0
    density_multiplier = 0.88623
    ev_speed = 593.097

    velocity_delta = ev_speed * math.sqrt(core_electron_energy_range)

    velocity_plus = velocity_delta + speed
    velocity_minus = velocity_delta - speed

    energy_plus = (velocity_plus / ev_speed) ** 2
    energy_minus = (velocity_minus / ev_speed) ** 2

    scalep2 = energy_multplier * energy_plus / temperature
    scalep = math.sqrt(scalep2)

    scalem2 = energy_multplier * energy_minus / temperature
    scalem = math.sqrt(scalem2)

    escalep = math.erfc(scalep) if scalep < 9 else 0
    escalem = math.erfc(scalem) if scalem < 9 else 0

    dscalep = density_multiplier / (scalep * np.exp(-scalep2) + density_multiplier * escalep)
    dscalem = density_multiplier / (scalem * np.exp(-scalem2) + density_multiplier * escalem)

    return 0.5 * (dscalep + dscalem)


def halotrunc(moments: Moments, core_halo_breakpoint: float, spacecraft_potential: float) -> Optional[np.float64]:
    htmax = moments.t_parallel
    htmin = moments.t_perpendicular

    speed = math.sqrt(moments.velocity_x ** 2 + moments.velocity_y ** 2 + moments.velocity_z ** 2)

    halod = moments.density

    if htmax > 1.0e8 or htmin > 1e8:
        return None

    temp = (htmax + 2 * htmin) / 3

    if htmax <= 1e4 or htmin <= 1e4:
        dscale = 1
    elif core_halo_breakpoint - spacecraft_potential > 5 and temp < 1.0e7:
        dscale = compute_density_scale(core_halo_breakpoint - spacecraft_potential, speed, temp)
    else:
        dscale = 1

    halod = halod / np.clip(dscale, 1, 5)

    return halod


def integrate(istart, iend, energy: np.ndarray, sintheta: np.ndarray,
              costheta: np.ndarray, deltheta: np.ndarray, fv: np.ndarray, phi: np.ndarray,
              spacecraft_potential: float, cdelnv: np.ndarray, cdelt: np.ndarray) -> Optional[
    CoreIntegrateOutputs]:
    # kmax? 30 or based on array size? expect different for different detectors?
    sumn = 0
    sumvx = 0
    sumvy = 0
    sumvz = 0
    base = 1000
    delphi = np.full(7, 2 * np.pi / 30)
    delv = np.empty((len(energy), 7))
    for i in range(istart, iend + 1):
        for j in range(7):
            if energy[i, j, 1] > 0:
                v2mid = (ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR ** 2) * energy[i, j, 1]
                v3mid = v2mid * np.sqrt(v2mid)
            else:
                v2mid = 0
                v3mid = 0
            ehigh = 1.175 * energy[i, j, 1]
            if i < len(energy) - 1:
                ehigh = np.sqrt((energy[i + 1, j, 1] + spacecraft_potential) * (energy[i, j, 1] + spacecraft_potential))
            elow = np.sqrt((energy[i, j, 1] + spacecraft_potential) * (energy[i - 1, j, 1] + spacecraft_potential))

            delv[i, j] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * (np.sqrt(ehigh) - np.sqrt(elow))
            if elow < base:
                base = elow

            for k in range(28):  # why 28 and not 30?
                if energy[i, j, k] > 0:
                    delta = delv[i, j] * deltheta[i, j] * delphi[j]
                    fact = sintheta[i, j] * fv[i, j, k]
                    sumn += delta * v2mid * fact
                    sumvx += delta * v3mid * fact * sintheta[i, j] * np.cos(np.deg2rad(phi[i, j, k]))
                    sumvy += delta * v3mid * fact * sintheta[i, j] * np.sin(np.deg2rad(phi[i, j, k]))
                    sumvz += delta * v3mid * fact * costheta[i, j]

    totden = sumn + cdelnv[0]
    if totden <= 0:
        return None
    eobolt = 12345

    base -= spacecraft_potential
    CM_PER_KM = METERS_PER_KILOMETER * CENTIMETERS_PER_METER
    KM_PER_CM = 1 / CM_PER_KM
    output_velocities = np.array([
        (-KM_PER_CM * sumvx + cdelnv[1]) / totden,
        (-KM_PER_CM * sumvy + cdelnv[2]) / totden,
        (-KM_PER_CM * sumvz + cdelnv[3]) / totden,
        base,
    ])

    sumtxx = 0
    sumtxy = 0
    sumtxz = 0
    sumtyy = 0
    sumtyz = 0
    sumtzz = 0
    sumqx = 0
    sumqy = 0
    sumqz = 0
    for i in range(istart, iend + 1):
        for j in range(7):
            v2mid = (ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR ** 2) * energy[i, j, 1]
            for k in range(28):  # again why 28?
                if energy[i, j, k] > 0:
                    angx = sintheta[i, j] * np.cos(np.deg2rad(phi[i, j, k]))
                    vx = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i, j, k]) * angx - CM_PER_KM * \
                         output_velocities[0]

                    angy = sintheta[i, j] * np.sin(np.deg2rad(phi[i, j, k]))
                    vy = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i, j, k]) * angy - CM_PER_KM * \
                         output_velocities[1]

                    angz = costheta[i, j]
                    vz = -ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[i, j, k]) * angz - CM_PER_KM * \
                         output_velocities[2]

                    vmag2 = vx * vx + vy * vy + vz * vz

                    delta = delv[i, j] * deltheta[i, j] * delphi[j]
                    fact = v2mid * sintheta[i, j] * fv[i, j, k]
                    sumtxx += delta * fact * vx * vx
                    sumtxy += delta * fact * vx * vy
                    sumtxz += delta * fact * vx * vz
                    sumtyy += delta * fact * vy * vy
                    sumtyz += delta * fact * vy * vz
                    sumtzz += delta * fact * vz * vz
                    sumqx += delta * fact * vmag2 * vx
                    sumqy += delta * fact * vmag2 * vy
                    sumqz += delta * fact * vmag2 * vz

    TEMPERATURE_SCALING_FACTOR_TO_UNDO_IN_EIGEN = 1e-4
    temperature = (np.array([sumtxx, sumtxy, sumtyy, sumtxz, sumtyz, sumtzz]) *
                   TEMPERATURE_SCALING_FACTOR_TO_UNDO_IN_EIGEN * eobolt + cdelt) / totden

    heat_flux = np.array([sumqx, sumqy, sumqz]) * 500 * ELECTRON_MASS_KG

    return CoreIntegrateOutputs(totden, output_velocities, temperature, heat_flux)


def scale_density(core_velocity: np.ndarray, core_temp: np.ndarray,
                  core_moment_fit: Moments, ifit: int, energy: np.ndarray,
                  spacecraft_potential: float, cosin_p: np.ndarray,
                  aperture_field_of_view: np.ndarray,
                  phi: np.ndarray,
                  regress_outputs: np.ndarray,
                  initial_core_density: float,
                  base_energy: float) -> ScaleDensityOutput:
    sumtxx = 0
    sumtxy = 0
    sumtyy = 0
    sumtxz = 0
    sumtyz = 0
    sumtzz = 0
    sumint = 0
    sumvx = 0
    sumvy = 0
    sumvz = 0

    eobolt = 12345

    zmk = ELECTRON_MASS_KG / (2 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN * 1e4)
    MAX_SPIN_SECTOR_INDEX = 28  # why 28 and not 30?
    KMAX = np.full(7, 30)

    exponent = np.full((2, NUMBER_OF_DETECTORS, MAX_SPIN_SECTOR_INDEX), -zmk * core_moment_fit.aoo)

    fv_integral = np.zeros((2, NUMBER_OF_DETECTORS, MAX_SPIN_SECTOR_INDEX))
    number_of_energies = 1
    assert 0 < ifit < len(energy) - 1, "ifit must be in middle of energies"
    delv = np.zeros((2, NUMBER_OF_DETECTORS))
    velocity_in_sc_frame = np.zeros((2, NUMBER_OF_DETECTORS))
    for j in range(NUMBER_OF_DETECTORS):
        ehigh = np.sqrt((energy[ifit + 1, j, 1] + spacecraft_potential) *
                        (energy[ifit, j, 1] + spacecraft_potential))
        elow = np.sqrt((energy[ifit, j, 1] + spacecraft_potential) *
                       (energy[ifit - 1, j, 1] + spacecraft_potential))
        delv[0, j] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * (np.sqrt(ehigh) - np.sqrt(elow))
        velocity_in_sc_frame[0, j] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(energy[ifit, j, 1])
        base_energy = min(base_energy, elow)
    base_energy -= spacecraft_potential
    if base_energy > 0:
        number_of_energies = 2
        velocity_in_sc_frame[1, :] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(0.5 * base_energy)
        delv[1, :] = ENERGY_EV_TO_SPEED_CM_PER_S_CONVERSION_FACTOR * np.sqrt(base_energy)
    factor = core_moment_fit.density * zmk / (np.pi * core_moment_fit.t_parallel) * np.sqrt(
        zmk / (np.pi * core_moment_fit.t_perpendicular))

    cos_theta = cosin_p
    sin_theta = np.sin(np.arccos(cos_theta))
    delta_theta = aperture_field_of_view
    cos_phi = np.cos(np.deg2rad(phi[0, 0]))
    sin_phi = np.sin(np.deg2rad(phi[0, 0]))
    delta_phi = 2 * np.pi / KMAX[-1]

    vscx = np.zeros((2, NUMBER_OF_DETECTORS, MAX_SPIN_SECTOR_INDEX))
    vscy = np.zeros((2, NUMBER_OF_DETECTORS, MAX_SPIN_SECTOR_INDEX))
    vscz = np.zeros((2, NUMBER_OF_DETECTORS, MAX_SPIN_SECTOR_INDEX))

    for j in range(NUMBER_OF_DETECTORS):
        for k in range(MAX_SPIN_SECTOR_INDEX):
            for i in range(number_of_energies):
                vx = -velocity_in_sc_frame[i, j] * sin_theta[j] * cos_phi[k]
                vy = -velocity_in_sc_frame[i, j] * sin_theta[j] * sin_phi[k]
                vz = -velocity_in_sc_frame[i, j] * cos_theta[j]
                vscx[i, j, k] = vx
                vscy[i, j, k] = vy
                vscz[i, j, k] = vz

                fun = -zmk * np.array([vx * vx, vy * vy, vz * vz, vx * vy, vx * vz, vy * vz, vx, vy, vz])
                exponent[i, j, k] += np.dot(fun, regress_outputs[:9])
                fv_integral[i, j, k] = factor * np.exp(exponent[i, j, k])

                delt = delv[i][j] * delta_theta[j] * delta_phi
                v2 = velocity_in_sc_frame[i][j] * velocity_in_sc_frame[i][j]

                common_factor = v2 * delt * sin_theta[j] * fv_integral[i, j, k]

                sumint += common_factor

                sumvx += vx * common_factor
                sumvy += vy * common_factor
                sumvz += vz * common_factor
                sumtxx += vx * vx * common_factor
                sumtxy += vx * vy * common_factor
                sumtxz += vx * vz * common_factor
                sumtyy += vy * vy * common_factor
                sumtyz += vy * vz * common_factor
                sumtzz += vz * vz * common_factor

    corrected_density = initial_core_density

    corrected_density += sumint

    corrected_core_velocity = core_velocity.copy()

    corrected_core_velocity *= initial_core_density / corrected_density
    corrected_core_velocity += np.array([sumvx, sumvy, sumvz]) * (-1e-5) / corrected_density

    sum_temperatures = np.array([sumtxx, sumtxy, sumtyy, sumtxz, sumtyz, sumtzz])
    corrected_core_temp = core_temp.copy()
    corrected_core_temp *= initial_core_density / corrected_density
    corrected_core_temp += sum_temperatures * 1e-4 * eobolt / corrected_density

    cdelnv = np.array([sumint, sumvx * (-1e-5), sumvy * (-1e-5), sumvz * (-1e-5)])

    cdelt = sum_temperatures * 1e-4 * eobolt

    return ScaleDensityOutput(density=corrected_density, velocity=corrected_core_velocity,
                              temperature=corrected_core_temp, cdelnv=cdelnv, cdelt=cdelt)
