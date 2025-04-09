source scripts/update_version.sh

if [[ "$1" == "swapi" ]]; then
python imap_l3_data_processor.py --instrument swapi --data-level l3a --start-date 20250606 \
--version v003 --dependency \
'[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v002.cdf"]}]'
python imap_l3_data_processor.py --instrument swapi --data-level l3b --start-date 20250606 \
--version v003 --dependency \
'[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v002.cdf"]}]'
elif [[ "$1" == "glows" ]]; then
python imap_l3_data_processor.py --instrument glows --data-level l3a --start-date 20150606 \
--version v004 --dependency \
'[{"type": "science", "files": ["imap_glows_l2_hist_20150606_v003.cdf"]}]'

fi

git restore imap_l3_processing/version.py
