docker run --rm --mount type=bind,src="$(pwd)/spice_kernels",dst="/mnt/spice" \
--mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" \
 $(docker build -q -f Dockerfile_run_local .) \
$@