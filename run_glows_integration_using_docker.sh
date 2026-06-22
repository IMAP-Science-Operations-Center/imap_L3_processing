source scripts/update_version.sh

#docker build -f Dockerfile_glows_integration .

docker run --rm \
--mount type=bind,src="$(pwd)/spice_kernels",dst="/spice_kernels" \
--mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" \
--mount type=bind,src="$(pwd)/run_local_input_data",dst="/run_local_input_data" \
--mount type=bind,src="$(pwd)/data",dst="/data" \
-e IMAP_API_KEY=$IMAP_API_KEY \
--entrypoint "" \
 $(docker build --platform linux/amd64 -q -f Dockerfile_glows_integration .) \
$@

git restore imap_l3_processing/version.py