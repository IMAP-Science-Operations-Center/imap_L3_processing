# IMAP L3 Data Processing
The level 3 processing code is a command-line program that will run inside of a docker container. 
The code snippet below will build the docker image and run the container and remove it after execution is completed.

## Running the Processor using SDC Infrastructure
### Uploading an L2 cdf file to trigger the L3 pipeline:
There is a program named `imap-data-access` which is used to upload and download cdf files from the SDC. 
Start by running the following commands in a terminal to create a virtual environment. If you've pulled the repository, open the terminal from the imap_l3_processing folder.
 - `python -m venv venv`
 - `source venv/Scripts/activate`

Continue following the installation instructions from the IMAP Science Operations Center. Installation instructions are found at: https://github.com/IMAP-Science-Operations-Center/imap-data-access

The SDC expects cdf files to follow a specific naming convention: `imap_swapi_{data_level}_{descriptor}_{start_date}_{version}.cdf`
An example file would be called: `imap_swapi_l2_sci_20111231_v001.cdf`

Note: File versions must be 3 characters long. (i.e. v003)

The command to upload a cdf file and trigger the pipeline is : `imap-data-access upload imap_swapi_l2_sci_20111231_v001.cdf`.
The command to see the newly created L3a data files is : `imap-data-access query --instrument swapi --data-level l3a --version v001`

You should see two L3a files:
* imap_swapi_l3a_proton-sw-fake-menlo-{GUID}_20111231_v001.cdf 
* imap_swapi_l3a_alpha-sw-fake-menlo-{GUID}_20111231_v001.cdf 


## Running the Processor Locally
### Setup
1. Install Docker Desktop
2. Start Docker Desktop
3. Pull this repository

### How to run the docker container from the command line
The arguments passed after the build command describe the inputs to the level 3 processing code.
The volume command-line argument mounts the local spice kernel folder to the docker container.
The remaining arguments match the inputs that we expect to receive from the SDC batch job run. 

`docker run --rm -it --volume="$(pwd)/spice_kernels:/mnt/spice" $(docker build -q .) --instrument swapi --level l3a --start_date 20240813 --end_date 20240813 --version v001 --dependency """[{'instrument':'swapi', 'data_level':'l2', 'descriptor':'fake-menlo-5-sweeps', 'version':'v001'}]"""`

Alternatively, running run_local_docker.sh in Git Bash will execute the above command for you:

`./run_local_docker.sh`

### Getting data from Dev data from SDC (Science Data Center)
We created a tool to retrieve the latest data from the SDC to assist with testing. The tool takes in command line arguements, the instrument level, and the number of files to retrieve. For example:

`python fetch_latest_data.py --instrument swapi --level l3a --count 4`

This will copy the .cdf files into your repo folder under the data folder. 

