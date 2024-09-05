docker run --rm -it --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .)  --instrument swapi --level l3a \
--start_date 20100101 --end_date 20100101 --version v002 --dependency \
"""[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'fake-menlo-5-sweeps', 'version':'v001'},
{'instrument':'swapi', 'data_level':'l2', 'descriptor':'density-temperature-lut-text-not-cdf', 'version':'v001'}]"""