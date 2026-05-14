#!/usr/bin/env python3
"""
Plot alpha peak-finding walkthrough on real L2 SWAPI spectra.

Three-panel figure (one per real-data fixture: strong alpha, hot plasma,
cold plasma) illustrating the alpha peak-finder and the subsequent Gaussian
fit on the count-rate residual:

  Panel top: 5-sweep-averaged observed count rate vs ESA voltage, overlaid
      with the frozen proton model and the detected alpha peak region shaded.
  Panel bottom: residual (observed − proton model) at the peak bins, with
      the fitted Gaussian.

Spectra are extracted from imap_swapi_l2_sci_20260101_v005.cdf; Stage 1
proton fits and rotation matrices are pre-computed in the test fixture at
tests/test_data/swapi/alpha_fit_test_spectra.npz.

Output: docs/swapi/figures/alpha_peak_finding.svg
Usage:  python docs/swapi/figure_src/plot_alpha_peak_finding.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.fit_solar_wind_alpha_model import (
    fit_solar_wind_alpha_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.fit_context import (
    build_solar_wind_fit_context,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.forward_model import (
    model_solar_wind_ideal_coincidence_rates,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.proton.fit_solar_wind_proton_model import (
    fit_solar_wind_proton_model,
)
from imap_l3_processing.swapi.l3a.science.solar_wind.params import SolarWindParams
from imap_l3_processing.swapi.response.deadtime import deadtime_factor
from imap_l3_processing.swapi.l3a.science.solar_wind.alpha.calculate_initial_guess import (
    _get_alpha_peak_indices,
)
from imap_l3_processing.swapi.l3a.utils import esa_voltage_to_alpha_speed
from imap_l3_processing.swapi.constants import SWAPI_K_FACTOR
from figure_utils import (
    COARSE_BIN_INDICES_IN_SWEEP,
    FIGURES_DIR,
    REPO_ROOT,
    compute_per_bin_rotation_matrices,
    load_swapi_response,
)

_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "test_data" / "swapi" / "alpha_fit_test_spectra.npz"
)

_N_SWEEPS = 5
_N_BINS = 62

_CASES = [
    ("strong_alpha", "Strong alpha (chunk 384)"),
    ("hot_plasma", "Hot plasma (chunk 250)"),
    ("cold_plasma", "Cold plasma (chunk 550)"),
]


def _load_fixture(data, name):
    prefix = f"{name}__"
    return {k[len(prefix) :]: data[k] for k in data.files if k.startswith(prefix)}


def _plot_case(axes_top, axes_bot, swapi_response, fixture, title):
    count_rates = fixture["count_rates"]
    voltage_per_sweep = fixture["voltage_per_sweep"]
    esa_flat = fixture["esa_flat"]
    proton_eff_scale = float(fixture["proton_eff_scale"])
    alpha_eff_scale = float(fixture["alpha_eff_scale"])
    magnetic_field_direction = fixture["b_hat_rtn"]
    cr_flat = fixture["cr_flat"]

    swapi_response.warm_cache(esa_flat)

    # Rebuild geometry to match the current forward model — the stored fixture
    # rotation matrices use an old convention.
    rotation_matrices = compute_per_bin_rotation_matrices(
        _N_SWEEPS, COARSE_BIN_INDICES_IN_SWEEP
    )

    proton_ctx = build_solar_wind_fit_context(
        count_rate=cr_flat,
        esa_voltage=esa_flat,
        swapi_response=swapi_response,
        central_effective_area_scale=proton_eff_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=PROTON_MASS_KG,
        mass_per_charge_m_p_per_e=PROTON_MASS_PER_CHARGE_M_P_PER_E,
    )
    proton_moments_obj = fit_solar_wind_proton_model(proton_ctx)
    proton_velocity_rtn = proton_moments_obj.bulk_velocity_rtn_nominal()
    proton_sw = SolarWindParams(
        density=proton_moments_obj.density.nominal_value,
        bulk_velocity_rtn=proton_velocity_rtn,
        temperature=proton_moments_obj.temperature.nominal_value,
        mass=PROTON_MASS_KG,
    )
    proton_true, _ = model_solar_wind_ideal_coincidence_rates(proton_sw, proton_ctx)
    proton_true_per_sweep = proton_true.reshape(_N_SWEEPS, _N_BINS)
    proton_obs_per_sweep = proton_true_per_sweep * deadtime_factor(
        proton_true_per_sweep
    )
    proton_bg_avg = proton_obs_per_sweep.mean(axis=0)
    count_avg = count_rates.mean(axis=0)

    alpha_ctx = build_solar_wind_fit_context(
        count_rate=cr_flat,
        esa_voltage=esa_flat,
        swapi_response=swapi_response,
        central_effective_area_scale=alpha_eff_scale,
        rotation_matrices=rotation_matrices,
        mass_kg=ALPHA_PARTICLE_MASS_KG,
        mass_per_charge_m_p_per_e=ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    )
    alpha_moments = fit_solar_wind_alpha_model(
        proton_ctx=proton_ctx,
        alpha_ctx=alpha_ctx,
        proton_moments=proton_moments_obj,
        magnetic_field_direction=magnetic_field_direction,
    )
    combined_fit_avg = None
    alpha_contribution_avg = None
    if np.isfinite(float(alpha_moments.density.nominal_value)):
        alpha_sw = SolarWindParams(
            density=alpha_moments.density.nominal_value,
            bulk_velocity_rtn=np.array(
                [c.nominal_value for c in alpha_moments.bulk_velocity_rtn]
            ),
            temperature=alpha_moments.temperature.nominal_value,
            mass=ALPHA_PARTICLE_MASS_KG,
        )
        alpha_true_fit, _ = model_solar_wind_ideal_coincidence_rates(
            alpha_sw, alpha_ctx
        )
        combined_true = proton_true + alpha_true_fit
        combined_fit = combined_true * deadtime_factor(combined_true)
        combined_fit_avg = combined_fit.reshape(_N_SWEEPS, _N_BINS).mean(axis=0)
        alpha_contribution_avg = np.maximum(combined_fit_avg - proton_bg_avg, 0.0)

    energies = SWAPI_K_FACTOR * np.abs(voltage_per_sweep)
    peak = _get_alpha_peak_indices(
        count_avg - proton_bg_avg, energies, count_avg.argmax()
    )
    peak_idx = np.arange(peak.start, peak.stop)
    residual_peak = np.maximum(count_avg[peak_idx] - proton_bg_avg[peak_idx], 0.0)
    speed_peak = esa_voltage_to_alpha_speed(voltage_per_sweep[peak_idx])
    has_peak = peak_idx.size > 0 and residual_peak.max(initial=0.0) > 0.0

    if has_peak:
        amplitude_guess = residual_peak.max()
        mean_guess = float(speed_peak[int(np.argmax(residual_peak))])
        try:
            (A_fit, mu_fit, sigma_fit), _ = scipy.optimize.curve_fit(
                lambda v, A, mu, sigma: A
                * np.exp(-((v - mu) ** 2) / (2 * sigma**2)),
                speed_peak,
                residual_peak,
                p0=[amplitude_guess, mean_guess, 50.0],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            )
        except RuntimeError:
            A_fit, mu_fit, sigma_fit = amplitude_guess, mean_guess, 50.0
    else:
        A_fit, mu_fit, sigma_fit = 0.0, 0.0, 1.0

    abs_voltage = np.abs(voltage_per_sweep)
    sort_idx = np.argsort(abs_voltage)
    abs_voltage_sorted = abs_voltage[sort_idx]

    axes_top.plot(
        abs_voltage_sorted,
        count_avg[sort_idx],
        ".",
        color="tab:blue",
        markersize=4,
        label="Observed (5-sweep avg)",
        zorder=3,
    )
    axes_top.plot(
        abs_voltage_sorted,
        proton_bg_avg[sort_idx],
        color="tab:orange",
        lw=1.5,
        label="Proton model",
        zorder=2,
    )
    if combined_fit_avg is not None:
        axes_top.plot(
            abs_voltage_sorted,
            combined_fit_avg[sort_idx],
            color="tab:purple",
            lw=1.5,
            linestyle="--",
            label="Combined fit (p + α)",
            zorder=2,
        )

    if has_peak:
        peak_voltages = abs_voltage[peak_idx]
        v_lo, v_hi = peak_voltages.min(), peak_voltages.max()
        axes_top.axvspan(
            v_lo, v_hi, alpha=0.15, color="tab:green", label="Alpha peak", zorder=1
        )

    axes_top.plot(
        abs_voltage[peak_idx],
        count_avg[peak_idx],
        "o",
        color="tab:green",
        markersize=5,
        markerfacecolor="none",
        lw=1.2,
        zorder=4,
    )

    axes_top.set_xscale("log")
    axes_top.set_yscale("log")
    axes_top.set_ylim(bottom=0.5)
    axes_top.set_ylabel("Count rate [Hz]")
    axes_top.set_title(title, fontsize=10)
    axes_top.legend(fontsize=7, loc="upper left")
    axes_top.grid(True, which="both", alpha=0.2)

    all_speeds = esa_voltage_to_alpha_speed(voltage_per_sweep)
    all_residual = np.maximum(count_avg - proton_bg_avg, 0.0)

    axes_bot.vlines(
        all_speeds[sort_idx],
        0,
        all_residual[sort_idx],
        colors="lightgrey",
        linewidth=1.5,
        zorder=1,
    )
    axes_bot.plot(
        all_speeds[sort_idx],
        all_residual[sort_idx],
        ".",
        color="grey",
        markersize=3,
        zorder=1,
        label="Residual (all bins)",
    )

    axes_bot.vlines(
        speed_peak, 0, residual_peak, colors="tab:green", linewidth=2, zorder=2
    )
    axes_bot.plot(
        speed_peak,
        residual_peak,
        "o",
        color="tab:green",
        markersize=5,
        zorder=2,
        label="Peak bins",
    )

    if has_peak:
        v_fine = np.linspace(speed_peak.min() - 50, speed_peak.max() + 50, 200)
        gauss_fine = A_fit * np.exp(-((v_fine - mu_fit) ** 2) / (2 * sigma_fit**2))
        axes_bot.plot(
            v_fine,
            gauss_fine,
            color="tab:red",
            lw=1.5,
            linestyle=":",
            label=rf"Gaussian (init. guess): $v_\alpha$={mu_fit:.0f} km/s",
            zorder=3,
        )

    if alpha_contribution_avg is not None:
        alpha_speed = np.linalg.norm(
            [c.nominal_value for c in alpha_moments.bulk_velocity_rtn]
        )
        axes_bot.plot(
            all_speeds[sort_idx],
            alpha_contribution_avg[sort_idx],
            color="tab:purple",
            lw=1.5,
            linestyle="--",
            label=rf"Moments fit: $v_\alpha$={alpha_speed:.0f} km/s",
            zorder=4,
        )

    axes_bot.set_xlabel(r"$\alpha$ speed [km/s]")
    axes_bot.set_ylabel("Residual [Hz]")
    axes_bot.legend(fontsize=7, loc="upper right")
    axes_bot.grid(True, alpha=0.2)
    y_top = max(residual_peak.max(initial=0.0), A_fit) * 1.3
    axes_bot.set_ylim(0, max(y_top, 10))


def main():
    print("Loading calibration data...")
    swapi_response = load_swapi_response()

    data = np.load(_FIXTURE_PATH)
    n_cases = len(_CASES)
    fig, axes = plt.subplots(
        2,
        n_cases,
        figsize=(4.5 * n_cases, 7),
        gridspec_kw={"height_ratios": [2, 1.2]},
    )
    fig.suptitle(
        r"Alpha peak-finding on real L2 spectra (imap\_swapi\_l2\_sci\_20260101)",
        fontsize=11,
    )

    for col, (case_name, case_title) in enumerate(_CASES):
        print(f"Plotting {case_name}...")
        fixture = _load_fixture(data, case_name)
        _plot_case(axes[0, col], axes[1, col], swapi_response, fixture, case_title)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "alpha_peak_finding.svg"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
