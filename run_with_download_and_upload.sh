source scripts/update_version.sh

if [[ "$1" == "swapi" ]]; then
python imap_l3_data_processor.py --instrument swapi --data-level l3a --start-date 20250606 \
--version v003 --dependency \
'[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v002.cdf"]}]'
python imap_l3_data_processor.py --instrument swapi --data-level l3b --start-date 20250606 \
--version v003 --dependency \
'[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v002.cdf"]}]'
elif [[ "$1" == "glows" ]]; then
python imap_l3_data_processor.py --instrument glows --data-level l3b --start-date 20100104 \
--version v004 --dependency \
'[{"type": "science", "files": ["imap_glows_l3a_hist_20100104_v002.cdf"]}]'
elif [[ "$1" == "hi" ]]; then
python imap_l3_data_processor.py --instrument hi --data-level l3 --start-date 20250415 \
--version v002 --descriptor "h90-sf-sp-hae-4deg-6mo" --dependency \
'[{"type": "science", "files": ["imap_hi_l2_h90-sf-ram-hae-4deg-6mo_20250415_v002.cdf"]},
{"type": "science", "files": ["imap_hi_l2_h90-sf-anti-hae-4deg-6mo_20250415_v002.cdf"]}]'
fi

git restore imap_l3_processing/version.py
