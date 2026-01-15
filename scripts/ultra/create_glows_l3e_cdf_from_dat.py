from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from imap_l3_processing.glows.descriptors import GLOWS_L3E_ULTRA_HF_DESCRIPTOR
from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments
from imap_l3_processing.glows.l3e.glows_l3e_ultra_model import GlowsL3EUltraData
from imap_l3_processing.models import InputMetadata
from imap_l3_processing.utils import save_data

version = 0

glows_dat_dir = Path(r'C:\Users\Harrison\Downloads\timeshifted_glows_data\fake_l3_validation')
glows_dat_files = list(glows_dat_dir.glob('*'))
output_dir = Path(r'C:\Users\Harrison\Downloads\timeshifted_glows_data\cdf')
output_dir.mkdir(exist_ok=True)

for glows_dat in glows_dat_files:
    full_date_str = glows_dat.name.split('_')[1]
    year_str, decimal_fractional_date_str = full_date_str.split('.')
    fractional_year = float(f'0.{decimal_fractional_date_str}')
    start_date = datetime(int(year_str), 1, 1) + 365 * timedelta(fractional_year)

    repointing = int(glows_dat.name.split('_')[2][:-4])

    input_metadata = InputMetadata(
        instrument="glows",
        data_level="l3e",
        descriptor=GLOWS_L3E_ULTRA_HF_DESCRIPTOR,
        start_date=start_date,
        end_date=start_date + timedelta(1),
        version=f"v{version:03}",
        repointing=repointing,
    )

    call_args_object = GlowsL3eCallArguments(
        formatted_date='',
        decimal_date='',
        spacecraft_radius=np.float32(0.0),
        spacecraft_longitude=np.float32(0.0),
        spacecraft_latitude=np.float32(0.0),
        spacecraft_velocity_x=np.float32(0.0),
        spacecraft_velocity_y=np.float32(0.0),
        spacecraft_velocity_z=np.float32(0.0),
        spin_axis_longitude=np.float32(0.0),
        spin_axis_latitude=np.float32(0.0),
        elongation=0,
    )

    glows_l3e = GlowsL3EUltraData.convert_dat_to_glows_l3e_ul_product(input_metadata,
        glows_dat,
        start_date,
        call_args_object,
    )

    ul_cdf = save_data(glows_l3e, folder_path=output_dir)