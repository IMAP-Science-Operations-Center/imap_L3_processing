FROM python:3.12-slim
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip
RUN mkdir temp_cdf_data
RUN mkdir -p /mnt/spice
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY spice_kernels spice_kernels
COPY imap_processing imap_processing
COPY imap_l3_data_processor.py .
ENTRYPOINT ["python","imap_l3_data_processor.py"]