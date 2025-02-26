import sys
from datetime import datetime

from spacepy.pycdf import CDF


def convert_epoch_time(filename, target=datetime(2025, 1, 1)):
    with CDF(str(filename), readonly=False) as cdf:
        delta = target - cdf['epoch'][0]
        cdf['epoch'][...] += delta


if __name__ == '__main__':
    convert_epoch_time(sys.argv[1])
