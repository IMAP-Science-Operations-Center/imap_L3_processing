docker run --rm --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .) --instrument glows --data-level l3a \
--start-date 20130908 --version v001 --dependency \
"""[{'instrument':'glows', 'data_level':'l2', 'descriptor':'hist', 'version':'v003', 'start_date':'20130908'}]"""