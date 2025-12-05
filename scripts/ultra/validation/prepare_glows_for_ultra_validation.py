import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

output_dir = Path(r"C:\Users\Harrison\Downloads\glows_l3e_for_ultra_validation")
output_dir.mkdir(exist_ok=True)
input_dir = Path(r"C:\Users\Harrison\Downloads\glows_l3e_from_validation\l3e")
for i, file in enumerate(input_dir.rglob('imap_glows_l3e_survival-probability-ul*.cdf')):
    if i > 184:
        break
    new_repoint = re.sub(r'repoint\d{5}', f'repoint{i:05}', file.name)
    new_epoch = datetime(2025,4,16)+timedelta(days=i)
    new_name = re.sub(r'\d{8}', new_epoch.strftime('%Y%m%d'), new_repoint)
    print(f'renaming {file.name} to {new_name}')
    shutil.copy(file, output_dir/new_name)

