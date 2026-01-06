import numpy as np
from matplotlib import pyplot as plt
from spacepy.pycdf import CDF

cdf = CDF("data/imap/swe/l2/2025/12/imap_swe_l2_sci_20251215_v005.cdf")
raw = cdf["phase_space_density"][...]
rebinned = np.where(raw == -1e31, np.nan, raw)

geometric_fractions = np.array([
    0.0781351,
    0.151448,
    0.204686,
    0.181759,
    0.175125,
    0.138312,
    0.0697327
])

rebinned_ma = np.ma.masked_invalid(rebinned)

rebinned_by_phi = np.ma.average(rebinned_ma, weights=geometric_fractions, axis=-1)
dist_1d = np.ma.average(rebinned_by_phi, axis=-1)
rebinned_by_theta = np.ma.average(rebinned_ma, axis=-2)

(fig, (ax1, ax2)) = plt.subplots(2, 1)
ax1.pcolormesh(dist_1d.filled(np.nan).T, norm="log")
ax2.pcolormesh(np.ma.mean(rebinned_by_phi[:, :, 0:4], axis=-1).filled(np.nan).T, norm="log")
plt.show()
