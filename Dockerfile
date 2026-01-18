FROM tensorflow/tensorflow:1.15.5-py3-jupyter

RUN pip install --no-cache-dir tqdm pytest

WORKDIR /app
