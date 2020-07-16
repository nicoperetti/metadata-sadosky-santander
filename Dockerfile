FROM anibali/pytorch:1.5.0-cuda10.2
USER root

RUN apt-get update && apt-get install build-essential -y

WORKDIR /meta
ENV PYTHONPATH /meta

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /meta

CMD ["/bin/bash"]