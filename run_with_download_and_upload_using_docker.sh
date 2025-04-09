source scripts/update_version.sh

docker run --rm --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .) --instrument glows --data-level l3a \
--start-date 20150606 --version v004 --dependency \
'[{"type": "science", "files": ["imap_glows_l2_hist_20150606_v003.cdf"]}]'


git restore imap_l3_processing/version.py