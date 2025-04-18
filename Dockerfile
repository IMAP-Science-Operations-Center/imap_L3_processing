FROM python:3.12-slim
RUN apt-get update && apt-get install -y git libgfortran5
RUN pip install --upgrade pip
RUN mkdir temp_cdf_data
RUN mkdir -p /mnt/spice
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY spice_kernels spice_kernels
COPY tests/test_data/fake_1_day_repointing_file.csv fake_1_day_repointing_file.csv
ENV REPOINT_DATA_FILEPATH=fake_1_day_repointing_file.csv
COPY imap_l3_processing imap_l3_processing
COPY imap_l3_data_processor.py .
ENTRYPOINT ["python","imap_l3_data_processor.py"]