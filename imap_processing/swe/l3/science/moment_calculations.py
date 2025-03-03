"""
int prep(int ifit, int nfit)
{

/* Sets up data for doing bi-Maxwellian fit to the core electrons */
/* Throw out points where fv=0, calculate velocity components */
/* Do this for nfit points starting with ifit. */

int i,j,k,ipt;
int ii,jj,kk;
int npts;
double v;

ipt = 0;
npts = 0;
for (j=0; j<N_CEMS; j++){
  npts += kmax[j]*nfit;
}

for (i=ifit; i<ifit+nfit; i++){
   for (j=0; j<N_CEMS; j++){
      for (k=0; k<kmax[j]; k++){
         if ((fv[i][j][k] > 0) && (energy[i][j][k] > 0)) {
            /* 5.93097e7 converts energy in eV to velocity in cm/s */
            v = 5.93097e7*sqrt(energy[i][j][k]);
            vxr[ipt] = -v*sintheta[i][j]*cos(phi[i][j][k]*pi/180);
            vyr[ipt] = -v*sintheta[i][j]*sin(phi[i][j][k]*pi/180);
            vzr[ipt] = -v*costheta[i][j];
            if (ipt == 51) {
            ii=i;
            jj=j;
            kk=k;
            }

            if (fv[i][j][k] > 1e-35)
               yreg[ipt] = log(fv[i][j][k]);
            else
               yreg[ipt] = -80.6;
            wait[ipt] = w[i][j][k];
            ipt++;
         }
         else
            npts--;
      }
   }
}

return ipt;

}
"""
import math
from dataclasses import dataclass

import numpy as np

from imap_processing.constants import ELECTRON_MASS_KG, \
    BOLTZMANN_CONSTANT_JOULES_PER_KELVIN

# CONSTANTS FOR COMPUTING MAXWELLIAN WEIGHTS
# Compute weight factors for bi-maxwellian fits.
#  sigma2 are variances due to digitization errors.
#  total variance includes both sigma2 and statistical variance (=counts)
LIMIT = np.array([32, 64, 128, 256, 1024, 2048, 3072, 5120, 9216, 17408, 33792])
SIGMA2 = np.array([0, 0.25, 1.25, 5.25, 21.25, 85.25, 341.25, 1365.25, 5461.25, 21845.25, 87381.25])
TSAMPLE = 1.0
MINIMUM_WEIGHT = 0.8165
MAX_VARIANCE = 349525.25

# CONSTANTS FOR COMPUTING MOMENTS WITH MAXWELL distribution (Units here are K / (m/s) ^2
# note the c code uses units of K / (cm/s)^2

ZMK = ELECTRON_MASS_KG / (2 * BOLTZMANN_CONSTANT_JOULES_PER_KELVIN)


def regress(self, velocity_vectors: np.ndarray[float], weight: np.ndarray[float], yreg: np.ndarray[float]) -> \
        np.ndarray[float]:
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

    y_fit = np.zeros(len(velocity_vectors))

    wt = np.zeros(len(velocity_vectors))

    array = np.zeros(shape=(9, 9))
    invarray = np.zeros(shape=(9, 9))

    sgmx = np.zeros(9)
    r = np.zeros(9)
    a = np.zeros(9)
    sa = np.zeros(9)

    xmean = np.zeros(9)
    ymean = 0
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
    sigma = np.sum(wt * yy * yy)

    for j in range(9):
        xx = fit_function[:, j] - xmean[j]
        sgmx[j] = np.sum(wt * xx * xx)
        r[j] = np.sum(wt * xx * yy)
        for k in range(j, 9):
            array[j][k] = np.sum(wt * xx * (fit_function[:, k] - xmean[k]))

    sigma = math.sqrt(sigma)
    sgmx = math.sqrt(sgmx)

    array = np.transpose(array)

    # TODO the C code has an error case here, the following line can throw LinAlgError which may or may not be what the C code handles
    invarray = np.linalg.inv(array)

    ao = ymean

    for j in range(9):
        a[j] = np.sum(r * invarray[j, :])
        ao -= a * xmean
        yfit = np.sum(a[j] * fit_function[:, j])

    a[9] = ao
    for i in range(len(velocity_vectors)):
        yfit[i] += ao;
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

    return a


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


# TODO figure out what the moments are and return a dataclass with them
def calculate_fit_temperature_density_velocity(parameters: np.array[float]):
    moments = Moments(alpha=0,
                      beta=0,
                      t_parallel=0,
                      t_perpendicular=0,
                      velocity_x=0,
                      velocity_y=0,
                      velocity_z=0,
                      density=0,
                      ao=0,
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

    vel = np.zeros(3)
    if d == 0:
        moments.velocity_x = 0
        moments.velocity_y = 0
        moments.velocity_z = 0
    else:
        moments.velocity_x = -1 * (parameters[6] * c11 + parameters[7] * c12 + parameters[8] * c13) / d
        moments.velocity_y = -1 * (parameters[6] * c12 + parameters[7] * c22 + parameters[8] * c23) / d
        moments.velocity_z = -1 * (parameters[6] * c13 + parameters[7] * c23 + parameters[8] * c33) / d

    fact = parameters[0] * moments.velocity_x * moments.velocity_x + parameters[
        1] * moments.velocity_y * moments.velocity_y + parameters[2] * \
           moments.velocity_z * moments.velocity_z
    moments.aoo = fact + parameters[3] * moments.velocity_x * moments.velocity_y + parameters[4] \
                  * moments.velocity_x * moments.velocity_z + \
                  parameters[5] * moments.velocity_y * moments.velocity_z

    moments.velocity_x = 1e-5
    moments.velocity_y = 1e-5
    moments.velocity_z = 1e-5

    if (moments.t_parallel < 0) or (moments.aoo < 0) or (moments.aoo > 1e13):
        moments.density = 0.0
    else:
        moments.density = (moments.t_perpendicular * np.sqrt(moments.t_parallel) / (
                (ZMK / math.pi) * np.sqrt(ZMK / math.pi))) * np.exp(
            moments.ao + ZMK * moments.aoo)
    return moments


# Compute weight factors for bi-maxwellian fits.
#  sigma2 are variances due to digitization errors.
#  total variance includes both sigma2 and statistical variance (=counts)
LIMIT = np.array([32, 64, 128, 256, 1024, 2048, 3072, 5120, 9216, 17408, 33792])
SIGMA2 = np.array([0, 0.25, 1.25, 5.25, 21.25, 85.25, 341.25, 1365.25, 5461.25, 21845.25, 87381.25])
TSAMPLE = 1.0
MINIMUM_WEIGHT = 0.8165
MAX_VARIANCE = 349525.25


# is corrected counts (ccounts) just count rate
def compute_maxwellian_weight_factors(corrected_counts: np.ndarray[float]) -> np.ndarray[float]:
    correction = 1.0 - 1.5e-6 * LIMIT / TSAMPLE
    correction[correction < 0.1] = 0.1
    xlimit = LIMIT / correction

    weights = np.empty_like(corrected_counts)

    for (energy_i, spin_i, declination_i), corrected_count in np.ndenumerate(corrected_counts):
        variance = MAX_VARIANCE
        for xlimit, sigma in zip((xlimit, SIGMA2)):
            if corrected_count < xlimit:
                variance = sigma

        if corrected_count <= 1.5:
            weights[energy_i, spin_i, declination_i] = MINIMUM_WEIGHT
        else:
            weights[energy_i, spin_i, declination_i] = np.sqrt(variance + corrected_count) / corrected_count

    return weights
