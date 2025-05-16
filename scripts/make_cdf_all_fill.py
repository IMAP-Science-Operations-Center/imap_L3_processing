import argparse
import shutil

import numpy as np
from spacepy.pycdf import CDF

parser = argparse.ArgumentParser()
parser.add_argument("template_cdf")

args = parser.parse_args()

output_file_name = args.template_cdf[:-4] + "-all-fill.cdf"
shutil.copy(args.template_cdf, output_file_name)
with CDF(output_file_name, readonly=False) as cdf:
    for var in cdf:
        if var not in ["Epoch", "epoch"] and cdf[var].attrs.get("VAR_TYPE") != "metadata":
            if "FILLVAL" in cdf[var].attrs:
                cdf[var] = np.full_like(cdf[var], cdf[var].attrs["FILLVAL"])
            else:
                print(f'The var {var} has no FILLVAL attribute.')
