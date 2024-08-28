# IMAP L3 Data Processing
The level 3 processing code is a command-line program that will run inside of a docker container. 
The code snippet below will build the docker image and run the container and remove it after execution is completed.

## Setup
1. Install Docker Desktop
2. Start Docker Desktop
3. Pull this repository

## How to run the docker container from the command line
The arguments passed after the build command describe the inputs to the level 3 processing code.
The volume command-line argument mounts the local spice kernel folder to the docker container.
The remaining arguments match the inputs that we expect to receive from the SDC batch job run. 

`docker run --rm -it --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .) --instrument swapi --level l3a --start_date 20240813 --end_date 20240813 --version v001 --dependency """[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'fake-menlo-5-sweeps', 'version':'v001'}]"""`

There is also a run_local_docker.sh that will execute the above command for you.