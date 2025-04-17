FROM python:3.12-slim
RUN apt-get -y update --fix-missing && apt-get -y install apt-utils
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install mcedit nano unzip curl wget openssh-client
RUN apt-get -y install make libgfortran5 unzip
# RUN apt-get -y install gfortran libgfortran-12-dev-amd64-cross
RUN apt-get -y install unzip
RUN mkdir /home/workdir
WORKDIR /home/workdir
COPY .inputrc /root/
COPY mc_bashrc /root/
RUN cd /root && cat mc_bashrc >> .bashrc
COPY Example.zip /home/workdir
RUN cd /home/workdir && unzip Example.zip
