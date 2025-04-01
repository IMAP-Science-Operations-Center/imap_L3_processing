from datetime import date

from spacepy.pycdf import CDF
import numpy as np

filename = f"imap_hi_l3_spectral-fit-index_{date.today().strftime("%Y%m%d")}_"
temp_cdf = CDF(
    rf"C:\Users\Harrison\Development\imap_L3_processing\temp_cdf_data\{filename}.cdf")
map_cdf = CDF(
    rf"C:\Users\Harrison\Development\cava\test\test_data\map_files\map_cdfs\{filename}.cdf")

map_cdf.readonly(False)
new_spectral_index_values = np.repeat(map_cdf['spectral_fit_index'][...][..., np.newaxis], len(map_cdf["bin"][...]),
                                      axis=-1)

# new_spectral_index_values = np.stack([map_cdf['spectral_fit_index'], map_cdf['spectral_fit_index'],
#                                       map_cdf['spectral_fit_index'], map_cdf['spectral_fit_index'],
#                                       map_cdf['spectral_fit_index']], axis=-1)
temp_cdf_attrs = temp_cdf['spectral_fit_index'].attrs
del map_cdf['spectral_fit_index']
map_cdf['spectral_fit_index'] = new_spectral_index_values
map_cdf['spectral_fit_index'].attrs = temp_cdf_attrs

map_cdf.close()
temp_cdf.close()
print("Finished")
