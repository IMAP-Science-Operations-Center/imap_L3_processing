docker run --rm --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .)  --instrument swapi --data-level l3a \
--start-date 20110210 --version v002 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'sci', 'version':'v001'}]"""