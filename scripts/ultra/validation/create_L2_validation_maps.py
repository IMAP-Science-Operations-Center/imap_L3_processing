import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path

from imap_data_access.file_validation import generate_imap_file_path
from imap_processing.cli import Ultra

OUTPUT_DIR = r"C:\Users\Harrison\Development\imap_L3_processing\generate_ultra_l2"

L2_DESCRIPTORS = [
    'u90-ena-h-hf-nsp-full-hae-2deg-2mo',
    'u90-ena-h-sf-nsp-full-hae-2deg-2mo',
    'u45-ena-h-hf-nsp-full-hae-2deg-2mo',
    'u45-ena-h-sf-nsp-full-hae-2deg-2mo',
    'u90-ena-h-hf-nsp-full-hae-4deg-2mo',
    'u90-ena-h-sf-nsp-full-hae-4deg-2mo',
    'u45-ena-h-hf-nsp-full-hae-4deg-2mo',
    'u45-ena-h-sf-nsp-full-hae-4deg-2mo',
    'u90-ena-h-hf-nsp-full-hae-6deg-2mo',
    'u90-ena-h-sf-nsp-full-hae-6deg-2mo',
    'u45-ena-h-hf-nsp-full-hae-6deg-2mo',
    'u45-ena-h-sf-nsp-full-hae-6deg-2mo',
]

VALIDATION_DESCRIPTORS = [
    'u90-ena-h-sf-sp-full-hae-2deg-3mo',
    'u90-ena-h-sf-sp-full-hae-4deg-3mo',
    'u90-ena-h-sf-sp-full-hae-6deg-3mo',
    'u45-ena-h-sf-sp-full-hae-2deg-3mo',
    'u45-ena-h-sf-sp-full-hae-4deg-3mo',
    'u45-ena-h-sf-sp-full-hae-6deg-3mo',
    'ulc-ena-h-sf-sp-full-hae-2deg-6mo',
]

DEPENDENCY_PATHS = [

]

def generate_l2_files():
    groups = defaultdict(list)
    for descriptor in L2_DESCRIPTORS:
        groups[descriptor[:15]].append(descriptor)
    with ProcessPoolExecutor() as pool:
        for group, descriptors in groups.items():
            pool.submit(make_maps, descriptors)
def make_maps(descriptors):
    print("making", descriptors)
    for descriptor in descriptors:
        sensor, _, _, frame, *_ = descriptor.split('-')
        dependency_filename = f"imap_ultra_l2_{sensor}-{frame}_20250929_v001.json"
        dependency = (Path(__file__).parent / "l2_dependency_files" / dependency_filename).read_text()

        version = "v000"
        start_date = '20250929'
        output_filename = f"imap_ultra_l2_{descriptor}_{start_date}_{version}.cdf"
        if not generate_imap_file_path(output_filename).construct_path().exists():
            Ultra("l2", descriptor, dependency, start_date, None, "v000", False).process()
        else:
            print(f"Skipping generation of {output_filename}, it already exists")

if __name__ == '__main__':
    generate_l2_files()

