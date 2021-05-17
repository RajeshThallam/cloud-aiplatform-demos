
FROM gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest
WORKDIR /pipeline
COPY train_examples.py train_examples.py
ENV PYTHONPATH="/pipeline:${PYTHONPATH}"
