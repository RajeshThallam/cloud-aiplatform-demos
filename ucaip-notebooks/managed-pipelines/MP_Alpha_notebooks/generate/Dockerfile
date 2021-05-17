
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest
WORKDIR /pipeline
COPY generate_examples.py generate_examples.py
RUN pip install tensorflow_datasets
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
