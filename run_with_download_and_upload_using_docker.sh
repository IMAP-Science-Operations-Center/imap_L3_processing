source scripts/update_version.sh

docker run --rm --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .) --instrument glows --data-level l3a \
--start-date 20150606 --version v003 --dependency \
"""[{'instrument':'glows', 'data_level':'l2', 'descriptor':'hist', 'version':'v003', 'start_date':'20150606'}]"""

git restore imap_l3_processing/version.py