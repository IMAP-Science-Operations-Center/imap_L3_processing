source scripts/update_version.sh

docker run --rm --mount type=bind,src="$(pwd)/spice_kernels",dst="/mnt/spice" \
--mount type=bind,src="$(pwd)/temp_cdf_data",dst="/temp_cdf_data" \
$(docker build -q .) --instrument glows --data-level l3b \
--start-date 20100104 --version v002 --dependency \
'[{"type": "science", "files": ["imap_glows_l3a_hist_20100104_v002.cdf"]}]'


git restore imap_l3_processing/version.py