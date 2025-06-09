FROM python:3.12-slim
RUN apt-get update && apt-get install -y git libgfortran5
RUN pip install --upgrade pip
RUN mkdir temp_cdf_data
RUN mkdir -p /mnt/spice
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY imap_l3_processing imap_l3_processing
COPY imap_l3_data_processor.py .
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbHi survProbHi
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbLo survProbLo
COPY imap_l3_processing/glows/l3e/l3e_toolkit/survProbUltra survProbUltra
RUN chmod +x survProbHi
RUN chmod +x survProbLo
RUN chmod +x survProbUltra
ENTRYPOINT ["python","imap_l3_data_processor.py"]