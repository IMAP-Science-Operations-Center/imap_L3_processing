import itertools
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imap_data_access

imap_data_access.config['DATA_ACCESS_URL']='https://api.dev.imap-mission.com/api-key'
imap_data_access.config['API_KEY']='e6378327cc4fe2d2eb35373a9b39de3f85a89f896bec3010e0476709868d1590'

l1_45_folder = Path(
    r"C:\Users\Harrison\Downloads\ULTRA_45sensor_spacecraft-psets_20251113-20251125T164844Z-1-001\ULTRA_45sensor_spacecraft-psets_20251113")
l1_90_folder = Path(
    r"C:\Users\Harrison\Downloads\ULTRA_90sensor_spacecraft-psets_20251113"
)
glows_folder = Path(
    r"C:\Users\Harrison\Downloads\ultra_combined_hf_validation\glows"
)
l2_45_folder = Path(r"C:\Users\Harrison\Downloads\u45_l2")
l2_90_folder = Path(r"C:\Users\Harrison\Downloads\u90_l2")
retries = 10

uploaded = []
failed = []
def upload_file(f):
    failures = 0
    while failures < retries:
        try:
            imap_data_access.upload(f)
            uploaded.append(f.name)
            print("uploaded", f.name)
            return
        except Exception as e:
            failures += 1

            print(f"failed to upload file: {f} after {failures} attempts, retrying...")
            if failures == retries:
                print(f"failed to upload file: {f} after {retries} attempts")
                failed.append(f)
                return
            time.sleep(failures ** 2 * 0.5)

with ThreadPoolExecutor() as executor:
    for f in itertools.chain(glows_folder.rglob("*.cdf")):
        f.rename(f.parent / f.name.replace("v021", "v022"))
        executor.submit(upload_file, f)

print("uploaded", uploaded)
print("failed", failed)


