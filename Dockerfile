FROM python:3.12-slim
RUN pip install --upgrade pip
RUN mkdir test_data
RUN mkdir -p /mnt/spice
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY swapi/swapi_l3a_sw_proton_speed.py swapi/.
COPY api_test.py .
COPY swapi/test_data/imap_swapi_l2_fake-menlo-5-sweeps_20100101_v001.cdf .
ENTRYPOINT ["python","api_test.py"]