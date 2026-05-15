## SWAPI He+ Pickup-Ion L3A Model

This document describes the SWAPI (Solar Wind and Pickup Ion) L3A helium pickup-ion (`pui-he`) model as implemented in `imap_l3_processing/swapi/l3a/science/pickup_ion`.
The data product is derived from fitting a filled-shell pickup-ion (PUI) distribution to SWAPI coincidence rate measurements.
For each 50-sweep chunk, roughly 10 minutes of data, the pipeline:

* selects the electrostatic-analyzer (ESA) voltage steps expected to contain the He+ PUI signal,
* precomputes the SWAPI response collapsed onto a solar-wind-frame speed grid,
* fits the PUI shell plus a flat coincidence-rate background with a Poisson likelihood,
* reports the fitted parameters, propagated density and temperature moments, and quality flags.

The main production entry point is `calculate_pickup_ion_values`; chunk-level orchestration is handled by `PuiChunkFitter`.

### Distribution Function

The model is a generalized filled-shell distribution for pickup ions at position $(r, \psi)$ in the solar inertial frame:
```math
f_\text{PUI}\!\left( r, \psi, w_k \right)
    = \frac{\alpha_{\text{PUI},k}}{4\pi}
      \cdot
      \frac{\beta_{E,k} r_E^2}{r u_\text{sw} v_{b,k}^3}
      \cdot
      w_k^{\alpha_{\text{PUI}, k} - 3}
      \cdot
      n_k (r w_k^{\alpha_{\text{PUI},k}}, \psi)
      \cdot
      \Theta(1 - w_k),
```
The free parameters are:

* Cutoff speed $v_{b,k}$.
* Cooling index $\alpha_{\text{PUI},k}$.
* Ionization rate $\beta_{E,k}$.
* Additive, energy-independent background coincidence rate.

The additional terms are:

* $`u_\text{sw}`$ is the solar wind speed in the inertial frame.
* $`w_k \equiv v'/v_{b,k}`$, where $`v' \equiv \|\mathbf{v} - \mathbf{v}_\text{sw}\|`$ is the speed in the solar-wind frame.
* $n(r, \psi)$ is the density of interstellar neutrals, precomputed into a lookup table using the hot model of Thomas (1978).
* $r_E \approx 1\,\text{au}$.

### Forward Model

SWAPI measures coincidence count rates as a function of ESA voltage.
For a voltage step $V$, the PUI contribution is the velocity distribution integrated with SWAPI's effective area function times speed across velocity space:
```math
C_\text{PUI}(V) = \int \text{d}^3v \thinspace  v \thinspace  f_\text{PUI}(\mathbf{v})  \mathcal{A}^{s}(\mathbf{v}, V).
```

$`\mathcal{A}^{s}`$ is specified in instrument coordinates ($`\theta, \phi, v/v_0(V)`$), so we must integrate over those coordinates using $`\mathrm{d}^3v = v^2 \cos\theta \, \mathrm{d} v \, \mathrm{d} \theta \, \mathrm{d} \phi`$:
```math
C_\text{PUI}(V) = \int \text{d}v \, \text{d}\phi \, \text{d}\theta \, v^3 \cos\theta f_\text{PUI}(\mathbf{v}) \mathcal{A}^{s}(\mathbf{v}, V).
```

For each $(\theta, \phi)$, we construct a unit vector $\hat{\mathbf{v}}(\theta, \phi)$ to specify $\mathbf{v} = v \hat{\mathbf{v}}$.
With $\mathbf{v}$ and $\mathbf{v}_\text{sw}$ provided in the same coordinate system and the model's free parameters specified, all variables needed for the model can then be calculated directly.
The production code instead precomputes a collapsed response so the fit loop does not repeatedly integrate over instrument angle.

### Coordinates and Notation

The integration uses SWAPI instrument coordinates $(\phi, \theta, \xi)$ where $\phi$ is azimuth, $\theta$ is elevation, and $\xi \equiv v/v_0(V)$.
The response tables are sampled in elevation and speed ratio, while azimuth enters through the SG (sunglasses) / OA (open aperture) response selection and azimuthal transmission table.

The velocity unit vector in SWAPI's Cartesian coordinate system is

```math
\hat{\mathbf{v}}(\theta, \phi)
  = \begin{pmatrix}
      -\cos\theta \sin\phi \\
      -\cos\theta \cos\phi \\
      -\sin\theta
    \end{pmatrix}
```

### Angular Reduction

This section explains the collapsed response used by `collapsed_response_grid.py`.
The purpose is to move the expensive angle-response integral out of the optimizer.

The Lorentz-invariant distribution $`f_\text{PUI}`$ is isotropic in the solar-wind frame. Changing variables $`\mathbf{u} \equiv \mathbf{v} - \mathbf{v}_\text{sw}`$ ($`\mathrm{d}^3v = \mathrm{d}^3u`$, $`v' = |\mathbf{u}|`$), using $`\mathrm{d}^3u = v'^{\,2}\,\mathrm{d}v'\,\mathrm{d}\Omega_u`$, and taking $`f_\text{PUI}(v')`$ out of the angular integral:

```math
C_\text{PUI}(V) = \int_0^\infty v'^{\,2}\,\mathrm{d}v' \; f_\text{PUI}(v') \, H(v', V),
```
where
```math
H(v', V) \;\equiv\; \int \mathrm{d}\Omega_{u} \; |\mathbf{v}|\, \mathcal{A}(\mathbf{v}, V)
```

is the angular integral of $`|\mathbf{v}|\,\mathcal{A}`$ on the spherical shell defined by $`|\mathbf{v}-\mathbf{v}_\text{sw}|=v'`$, with dimensions of volume per time.
We can convert the angular integral to an integral in velocity space over the shell as follows:
```math
H(v', V) = \int \mathrm{d}^3 v \; \frac{\delta(v' - \| \mathbf{v} - \mathbf{v}_\text{sw} \|)}{v'^2} \, |\mathbf{v}|\, \mathcal{A}(\mathbf{v}, V)
```

In spherical coordinates:
```math
H(v', V) = \int \mathrm{d}v \int \mathrm{d}\theta \int \mathrm{d}\phi \; v^3 \cos\theta \; \frac{\delta(v' - \| \mathbf{v} - \mathbf{v}_\text{sw} \|)}{v'^2} \, \mathcal{A}(\mathbf{v}, V)
```

To make this integral easier to calculate, we can fix $v'$ and integrate over two coordinates from $(\theta, \phi, v)$.
We choose to integrate over $(\theta, v)$ so that we only need to interpolate the passband once for each region.

The term inside the delta function is satisfied when:
```math
v'^2
= v^2 + v_\text{sw}^2 - 2 v v_\text{sw} \cos\alpha,
```
where $\alpha$ denotes the angle between $\mathbf{v}$ and $\mathbf{v}_\text{sw}$, given by:
```math
\cos\alpha = \frac{v^2 + v_\text{sw}^2 - v'^2}{2 v v_\text{sw}}
= \sin\theta \sin \theta_b + \cos\theta \cos\theta_b \cos(\phi - \phi_b).
```

Solving for $\phi$,
```math
\cos(\phi - \phi_b) = \frac{\cos\alpha - \sin\theta \sin\theta_b}{\cos\theta \cos\theta_b},
```
which has zero or two solutions (depending on whether the right-hand side lies in $[-1, 1]$):
```math
\phi_\pm = \phi_b \pm \arccos\!\left( \frac{\cos\alpha - \sin\theta \sin\theta_b}{\cos\theta \cos\theta_b} \right)
```
(The boundary case where the argument equals $\pm 1$ yields a single (degenerate) root $\phi_+ = \phi_-$ but can be ignored for integration purposes.)

Define $`R(\phi) \equiv \|\mathbf{v} - \mathbf{v}_\text{sw}\|`$ at fixed $(v, \theta)$. Differentiating $`R^2(\phi) = v^2 + v_\text{sw}^2 - 2 v v_\text{sw} \cos\alpha(\phi)`$ with respect to $\phi$ gives
```math
2 R(\phi)\, R'(\phi) = 2\, v\, v_\text{sw} \cos\theta \cos\theta_b \sin(\phi - \phi_b),
```
so
```math
R'(\phi) = \frac{v\, v_\text{sw} \cos\theta \cos\theta_b \sin(\phi - \phi_b)}{R(\phi)},
\qquad
|R'(\phi_\pm)| = \frac{v\, v_\text{sw} \cos\theta \cos\theta_b\, |\sin(\phi_\pm - \phi_b)|}{v'}
```
at the roots, where $R(\phi_\pm) = v'$.

The standard identity $`\delta(v' - R(\phi)) = \sum_\pm \delta(\phi - \phi_\pm) / |R'(\phi_\pm)|`$ collapses the $\phi$ integral:
```math
H(v', V) = \int \mathrm{d}v \int \mathrm{d}\theta \; \frac{v^3 \cos\theta}{v'^{\,2}} \sum_\pm \frac{\mathcal{A}(v, \theta, \phi_\pm, V)}{|R'(\phi_\pm)|}.
```
Substituting $|R'(\phi_\pm)|$ cancels the $\cos\theta$ from $`\mathrm{d}^3v = v^2 \cos\theta\,\mathrm{d}v\,\mathrm{d}\theta\,\mathrm{d}\phi`$ against the one in $|R'|$, and one power of $v$ and $v'$ each:
```math
H(v', V) = \int \mathrm{d}v \int \mathrm{d}\theta \; \frac{v^2}{v_\text{sw}\, v' \cos\theta_b} \sum_\pm \frac{\mathcal{A}(v, \theta, \phi_\pm, V)}{|\sin(\phi_\pm - \phi_b)|},
```
with the sum running over real, non-degenerate roots $\phi_\pm$. The remaining integrand depends on $\mathbf{v}_\text{sw}$ only through $(v_\text{sw}, \theta_b, \phi_b)$ and on the response only through $\mathcal{A}$ at the two roots, both of which are bounded on the tabulated grid; the $1/|\sin(\phi_\pm - \phi_b)|$ factor is integrable across each cell.

Factoring the response as $\mathcal{A}(v, \theta, \phi, V) = A_0(V)\, P(\theta, v/v_0;\, \text{region}(\phi))\, T(\phi)$, the implementation evaluates a 32 × 32 quadrature over elevation $\theta \in [-15^\circ, 15^\circ]$ and speed ratio $v/v_0 \in [0.9, 1.1]$, skips cells where both passband regions are below $10^{-3}$, and adds the following contribution after summing over real roots:
```math
\Delta H(v', V) =
\Delta v\,\Delta\theta\,
\frac{v^2 A_0(V)}{v_\text{sw} v' \cos\theta_b \sqrt{1 - \cos^2(\phi - \phi_b)}}
\sum_\pm P_\pm(\theta, v/v_0)\,T(\phi_\pm),
```
where $A_0(V)$ is the central effective area converted to $\mathrm{km}^2$, $P_\pm$ is the SG passband for $|\phi_\pm| \leq 20^\circ$ and the OA passband otherwise, and $T$ is the azimuthal transmission.
The sum runs only over real, non-degenerate roots of $\phi_\pm$.
Near-degenerate roots with $|\cos(\phi-\phi_b)| \approx 1$ are skipped in code to avoid a numerical spike from the $1/\sin(\phi-\phi_b)$ factor; the excluded boundary has zero measure in the continuous integral.

![Closed-form φ-inversion result compared with a numerical shell-integral reference for H(v', V) at V=5000 V, m/q=4, bulk=450 km/s at (az=5°, el=−10°)](figures/collapsed_response_grid.svg)

*Generated by `docs/swapi/figure_src/plot_collapsed_response_grid.py`.*

### Solar-Wind-Frame Speed Support

For one response grid at central speed $v_0$, $H(v', V)$ is nonzero only where the response passband and the shell $|\mathbf{v}-\mathbf{v}_\text{sw}|=v'$ overlap.
That support is constrained by the following:

1. SWAPI's energy-angle passband, taken quite conservatively:
```math
0.9\, v_0 < v < 1.1\, v_0
```
2. The range of possible $v'$ for a given $v$:
```math
v - v_\text{sw} \leq v' \leq v + v_\text{sw}
```

For the upper bound, it is straightforward:
```math
v'_\text{max} = 1.1\, v_0 + v_\text{sw}.
```

For the lower bound, there are several possibilities to consider:
- If $v_\text{sw} < 0.9\, v_0$, then $v'_\text{min} = 0.9\, v_0 - v_\text{sw}$.
- If $0.9 v_0 \leq v_\text{sw} \leq 1.1 v_0$, then $v'_\text{min} = 0$.
- If $v_\text{sw} > 1.1\, v_0$, then $v'_\text{min} = v_\text{sw} - 1.1\, v_0$.

Equivalently:
```math
v'_\text{min} = \max\!\left(0.9\, v_0 - v_\text{sw},\; 0,\; v_\text{sw} - 1.1\, v_0\right)
```

The width of this support interval is then
```math
v'_\text{max} - v'_\text{min} = \min\!\left(0.2\, v_0 + 2 v_\text{sw},\; 1.1\, v_0 + v_\text{sw},\; 2.2\, v_0\right).
```
The three arguments correspond to $v_\text{sw} < 0.9\, v_0$, $v_\text{sw} \in [0.9\, v_0, 1.1\, v_0]$, and $v_\text{sw} > 1.1\, v_0$, respectively.
In code, `solar_wind_frame_speed_range` returns these support bounds, and `build_collapsed_response_grid` evaluates $H$ only at shared-grid points inside them.

### Numerical Integration Scheme

This is the main performance trick in the PUI fit.
The response-dependent weights are precomputed once per chunk; each optimizer evaluation only evaluates the distribution on a 1D speed grid and performs a tensor contraction.

`build_chunk_collapsed_response` precomputes one shared 256-point $v'$ grid per chunk.
Let $u_\text{fit}$ be the upstream proton speed passed to `fit_pickup_ion_parameters`, and let $r_\text{LUT,min}$ be the minimum radius in the neutral-helium LUT, in au.
The grid is
```math
\begin{aligned}
v'_0 &= \max\!\left(1\,\mathrm{km/s},\; 0.8\,u_\text{fit}\,\frac{r_\text{LUT,min}}{r_\text{au}}\right), \\
\Delta v' &= \frac{1.2\,u_\text{fit} - v'_0}{256 - 1.5}, \\
v'_i &= v'_0 + i\,\Delta v', \qquad i=0,\ldots,255.
\end{aligned}
```
This places the maximum fit-allowed cutoff speed at the left edge of the final grid cell.

For each sweep and voltage step, the precomputed weight is
```math
W_{s,k,i} = H_{s,k}(v'_i, V_k)\, v_i'^2\,q_i,
```
where $q_i$ is the trapezoidal integration width: $\Delta v'/2$ for the first and last grid points and $\Delta v'$ otherwise.
The forward model samples $f_\text{PUI}$ at the fixed grid centers $v'_i$.
The distribution's Heaviside factor zeros grid centers above the cutoff.
The one grid point selected by
```math
j = \left\lfloor \frac{v_\text{cut} - v'_0}{\Delta v'} \right\rfloor
```
receives the production partial-cell correction, when $0 \leq j < 255$:
```math
\tilde f_j =
\frac{v_\text{cut} - (v'_j - \Delta v'/2)}{\Delta v'}\, f_\text{PUI}(v'_j),
\qquad
\tilde f_i = f_\text{PUI}(v'_i)\quad (i\neq j).
```
The implementation does not shift the sample location to the midpoint of the partial sub-interval; it keeps all distribution samples on the fixed shared grid.

```math
C_\text{model}(V_k) \;\approx\; \sum_i W_{s,k,i}\,\tilde f_i + C_\text{bg}.
```

$W_{s,k,i}$ depends on the instrument response and the per-bin bulk solar-wind velocity in SWAPI coordinates, but not on the PUI fit parameters. We precompute $W_{s,k,i}$ once per 50-sweep chunk and reuse it across every fit iteration. Each iteration samples $f_\text{PUI}$ at the grid centers, applies the cutoff-cell correction, and sums against $W$ to recover the model count rate for every sweep and ESA voltage in the chunk — a single matrix–vector product in code plus the fitted flat background.

### Moment Integrals

After fitting, `moments.py` evaluates density and temperature from the fitted distribution rather than from the instrument response.
These moments use the same speed grid and cutoff correction as the forward model so the reported moments are consistent with the fitted parameters.

The PUI number density and temperature are speed moments of $f_\text{PUI}$ in the solar-wind frame:
```math
n_\text{PUI} = 4\pi \int_0^{v_\text{cut}} v'^{\,2} \, f_\text{PUI}(v') \; \mathrm{d}v',
\qquad
T_\text{PUI} = \frac{m}{3 k_B}
  \frac{\int_0^{v_\text{cut}} v'^{\,4} \, f_\text{PUI}(v') \; \mathrm{d}v'}
       {\int_0^{v_\text{cut}} v'^{\,2} \, f_\text{PUI}(v') \; \mathrm{d}v'}.
```

We evaluate these on the same $v'$ grid the forward model uses, with the same trapezoidal widths $q_i$ and the same cutoff-cell correction. Stripping the instrument-response factor $H(v', V)$ out of $W_{s,k,i}$ leaves the purely geometric per-bin weight
```math
I_i \;\equiv\; v'_i{}^2 \, q_i.
```

Because the implementation evaluates speeds in km/s and $f_\text{PUI}$ in km-based units, the reported density is converted from $\mathrm{km}^{-3}$ to $\mathrm{cm}^{-3}$ and the temperature calculation converts $(\mathrm{km/s})^2$ to $(\mathrm{m/s})^2$:
```math
\begin{aligned}
n_\text{PUI} &\;\approx\; \frac{4\pi}{(10^5)^3} \sum_i I_i \, \tilde f_i, \\
T_\text{PUI} &\;\approx\; \frac{m}{3 k_B}
  (10^3)^2
  \frac{\sum_i v'_i{}^2 \, I_i \, \tilde f_i}
       {\sum_i I_i \, \tilde f_i}.
\end{aligned}
```

The lower edge $v'_0$ is chosen from the smallest fit-allowed cutoff speed and the LUT minimum radius so speeds below the grid are in the LUT-zero-density region for fit-allowed $(\alpha, v_b)$; the 1 km/s floor avoids evaluating the negative power of $w$ at zero.

### Cross-Validation Against a Reference Integral

Two references keep the implementation anchored:
`test_build_collapsed_response_grid.py` compares the collapsed angular response against a direct numerical shell integral, and `test_calculate_coincidence_rate.py` compares the production forward model against an xarray + Pint reference.

![xarray + pint reference: total coincidence-rate spectrum across the ESA voltage table, using parameters specified in the reference script.](figures/pui_xarray_reference.svg)

*Generated by `docs/swapi/figure_src/plot_pui_reference_comparison.py` from `tests/test_data/swapi/pui_count_rate_reference.csv`.*

## Fitting Routine

The fitter consumes a chunk of coincidence-rate measurements — 50 sweeps × 62 coarse ESA voltage steps — together with the upstream proton bulk velocity. The chunk-average RTN velocity sets the cutoff bounds and Vasyliunas–Siscoe distribution geometry; the per-bin SWAPI-frame velocity vectors set the collapsed instrument response. The fit returns nominal values plus 1σ uncertainties for the four free parameters; `PuiChunkFitter` then computes density and temperature moments and combines quality flags.

### Bin Selection

Only ESA voltage steps in the He+ PUI band are fit. The implementation averages voltage over the 50 sweeps at each coarse step, converts that voltage to L2 energy using `SWAPI_L2_K_FACTOR`, and keeps a step if and only if
```math
1.25 \, E_\text{cut}(\mathrm{H^+}) \;<\; kV \;<\; 1.2 \, E_\text{cut}(\mathrm{He^+}),
```
where $E_\text{cut}(s)$ is the kinematic PUI cutoff energy for species $s$ — the kinetic energy in the SWAPI frame of an ion born at rest in the inflow gas and convected at the bulk solar-wind velocity. The lower edge sits above the H+ PUI cutoff to suppress proton-PUI contamination of the residual; the upper edge slightly overshoots $E_\text{cut}(\mathrm{He^+})$ so the cutoff itself falls inside the fit window rather than on its boundary.
The cutoff energies are computed in `calculate_pui_energy_cutoff`, using the H and He inflow vectors and the chunk-center ephemeris time.

### Bounds and Initial Simplex

The four free parameters are fit with hard bounds:

| Parameter | Lower | Upper | Initial |
|---|---|---|---|
| $\alpha_\text{PUI}$ | 1.0 | 5.0 | 1.5 |
| $\beta_E$ | $0.6 \times 10^{-9}$ s⁻¹ | $8 \times 10^{-7}$ s⁻¹ | $10^{-7}$ s⁻¹ |
| $v_b$ | $0.8 \, u_\text{fit}$ | $1.2 \, u_\text{fit}$ | $u_\text{fit}$ |
| Background | 0 | 10 Hz | 0.1 Hz |

Here $u_\text{fit} = \|\mathbf{v}_\text{sw,RTN}\|$ is the upstream proton speed passed into the fitter; it is distinct from the inertial-frame $u_\text{sw}$ used in the Vasyliunas–Siscoe normalization.

An explicit five-vertex initial simplex perturbs one parameter at a time away from the initial point ($\alpha_\text{PUI} \to 5.0$, $\beta_E \to 2.1 \times 10^{-7}$ s⁻¹, $v_b \to 1.2 \, u_\text{fit}$, background $\to 0.2$ Hz), so each parameter has a non-degenerate starting span.

### Likelihood

The forward model produces a predicted count rate $C_\text{model}(V_k; \theta)$ for every (sweep $s$, voltage step $k$) in the chunk, including the PUI shell and fitted flat background. Converting to expected counts using the SWAPI per-step livetime $\tau_\text{lt} = 0.145$ s,
```math
m_{s,k}(\theta) = \tau_\text{lt} \cdot C_\text{model}(V_k; \theta),
\qquad
n_{s,k} = \tau_\text{lt} \cdot r_{s,k},
```
where $r_{s,k}$ is the observed L2 coincidence rate. Assuming independent Poisson statistics per bin, the negative log-likelihood is
```math
\mathcal{L}(\theta) \;=\; \sum_{s,k} \bigl[ m_{s,k}(\theta) - n_{s,k} \, \ln m_{s,k}(\theta) \bigr],
```
dropping the parameter-independent $\ln n_{s,k}!$ term. No explicit epsilon is added before the logarithm; valid fit evaluations require $m_{s,k} > 0$.

### Optimizer

Optimization uses the Nelder–Mead simplex via `lmfit`.
Nelder–Mead is used here because the forward model has a cutoff correction that is not smoothly differentiable at every $v'$ grid edge.
Bounded parameters are mapped to unbounded internal coordinates by the arcsine transform
```math
x_\text{int} \;=\; \arcsin\!\bigl( 2 (x - x_\text{min}) / (x_\text{max} - x_\text{min}) - 1 \bigr),
```
so the simplex traverses an unconstrained space while bounds remain honored at the external $x$.

### Uncertainty Estimation

Asymptotic maximum-likelihood covariance is taken from the inverse Hessian of $\mathcal{L}$ at the optimum in internal coordinates:
```math
\Sigma_\text{int} = \bigl[ \nabla^2 \mathcal{L}(\hat\theta_\text{int}) \bigr]^{-1},
\qquad
\Sigma_\text{ext} = J \, \Sigma_\text{int} \, J^\top,
```
where $J$ is the Jacobian of the inverse arcsine transform at $\hat\theta_\text{int}$. Per-parameter standard errors are $\hat\sigma_p = \sqrt{[\Sigma_\text{ext}]_{pp}}$. The finite-difference step for the Hessian is deliberately coarse (1e-2 in internal coordinates): the partial-Heaviside cutoff correction makes $\mathcal{L}$ piecewise linear in $v_b$ across each $v'$-bin edge, and a finer step would resolve sub-bin local curvature that misrepresents the parameter-scale Hessian.

### Quality Flags

A chunk is flagged `BAD_FIT` if either:

- One or more standard errors $\hat\sigma_p$ are non-finite — $\Sigma_\text{int}$ was not positive definite, typically because a parameter was pulled to a bound where the asymptotic-MLE approximation breaks down.
- The model–data coefficient of determination $R^2 = 1 - \mathrm{SS}_\text{res} / \mathrm{SS}_\text{tot}$, computed on count rates over all fitted bins in the chunk, falls below 0.9 (or is NaN).

In either `BAD_FIT` branch, all four fitted parameters are returned as NaN ± NaN and the helium-PUI density and temperature derived from them are NaN as well, matching the proton- and alpha-SW behavior on bad fits.

When the fit is good, an additional per-parameter fill rule applies: if the fitted background exceeds 1 Hz, the flat term is absorbing real signal rather than measuring SWAPI's coincidence baseline; the background is reported as NaN ± NaN while the other three parameters are kept as fit. This rule does not set `BAD_FIT`.

`PuiChunkFitter` then bitwise-ORs this PUI fit flag with the upstream proton-SW quality flag. If the input count rates or upstream velocity arrays contain NaNs, the PUI fit is skipped and all PUI outputs are returned as NaN ± NaN with only the upstream flag propagated.

### Monte Carlo Parameter Recovery

We run the fit against a 50-sweep × 62-step coincidence-rate fixture
(`tests/test_data/swapi/pui_count_rate_reference_50sweep.h5`) that sums the
reference PUI integral, a proton + alpha Maxwellian shoulder, and a flat
background, with the deadtime factor applied. Each MC realization samples the
resulting expected counts from a Poisson distribution and sends the chunk
through `ParallelChunkRunner` and the production PUI fit path. The upstream
solar-wind velocity, per-bin SWAPI-frame bulk vectors, and proton + alpha
shoulder are supplied by the fixture rather than re-fit in each realization;
the MC test uses `_FitOnlyPuiChunkFitter` to skip density/temperature moment
post-processing and validate only the four fit parameters.
The fitted-model spectrogram below is reconstructed by forward-modeling the
fitted PUI shell, adding the fixture's proton + alpha Maxwellian shoulder and
the fitted flat background, and applying the deadtime factor — directly
comparable to the truth spectrogram.

![PUI fit MC validation: truth model, Poisson realization, and fitted total model spectrograms (shared log color scale), followed by per-parameter histograms of fitted nominal values, reported σ̂, and `(fit − truth) / σ̂` over 1000 Poisson realizations.](figures/pui_mc_validation.svg)

*Generated by `docs/swapi/figure_src/plot_pui_mc_validation.py` from `tests/test_data/swapi/pui_count_rate_reference_50sweep.h5`.*
