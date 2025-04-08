source scripts/update_version.sh

if [[ "$1" == "swapi" ]]; then
python imap_l3_data_processor.py --instrument swapi --data-level l3a --start-date 20250606 \
--version v003 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'sci', 'version':'v002', 'start_date':'20250606'}]"""
python imap_l3_data_processor.py --instrument swapi --data-level l3b --start-date 20250606 \
--version v003 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'sci', 'version':'v002', 'start_date':'20250606'}]"""
elif [[ "$1" == "glows" ]]; then
python imap_l3_data_processor.py --instrument glows --data-level l3a --start-date 20130908 \
--version v020 --dependency \
"""[{'instrument':'glows', 'data_level':'l2', 'descriptor':'histogram-00001', 'version':'v003', 'start_date':'20130908'}]"""
fi

git restore imap_l3_processing/version.py
