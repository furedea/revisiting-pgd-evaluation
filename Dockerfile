FROM tensorflow/tensorflow:1.15.5-gpu-py3-jupyter

WORKDIR /workspace

RUN pip install --no-cache-dir pytest tqdm

COPY . /workspace/

CMD ["pytest", "tests/", "-v"]
