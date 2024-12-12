if [[ "$1" == "swapi" ]]; then
python imap_l3_data_processor.py --instrument swapi --data-level l3a --start-date 20250606 \
--version v003 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'sci', 'version':'v002'}]"""
python imap_l3_data_processor.py --instrument swapi --data-level l3b --start-date 20250606 \
--version v003 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'sci', 'version':'v002'}]"""
elif [[ "$1" == "glows" ]]; then
python imap_l3_data_processor.py --instrument glows --data-level l3a --start-date 20130908 \
--version v020 --dependency \
"""[{'instrument':'glows', 'data_level':'l2', 'descriptor':'histogram-00001', 'version':'v003'}]"""
fi
