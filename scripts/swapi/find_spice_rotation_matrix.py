import datetime

import numpy as np
import spiceypy


def find_spice_rotation_matrix():
    spiceypy.furnsh(['data/imap/spice/spk/imap_recon_20250415_20260415_v01.bsp', 'data/imap/spice/spk/de440.bsp',
                     'data/imap/spice/ck/imap_2025_105_2026_105_01.ah.bc',
                     'data/imap/spice/ck/imap_dps_2025_105_2026_105_01.ah.bc',
                     'data/imap/spice/fk/imap_science_0001.tf',
                     'data/imap/spice/fk/imap_001.tf',
                     'data/imap/spice/sclk/imap_sclk_0000.tsc',
                     'data/imap/spice/lsk/naif0012.tls'])

    et = spiceypy.datetime2et(datetime.datetime(year=2025, month=6, day=21, hour=12))

    position = spiceypy.spkezr("IMAP", et, 'ECLIPJ2000', 'NONE', 'SUN')[0][0:3]
    velocity = spiceypy.spkezr("IMAP", et, 'ECLIPJ2000', 'NONE', 'SUN')[0][3:6]
    print(position, velocity)
    distance_km, longitude, latitude = spiceypy.reclat(position)

    def get_matrix(t):
        return np.round(spiceypy.sxform('IMAP_SWAPI', 'ECLIPJ2000', t)[3:6,3:6], decimals=2)
    print(np.round(spiceypy.sxform('IMAP_DPS', 'ECLIPJ2000', et), decimals=0))
    print(get_matrix(et+11.7))

if __name__ == '__main__':
    find_spice_rotation_matrix()

