FROM tensorflow/tensorflow:2.2.0rc1-gpu-jupyter

RUN DEV_PACKAGES=" \
        git \
 " \
 && apt-get update \
 && apt-get install -y \
        graphviz \
 && apt-get install -y $DEV_PACKAGES \
 && pip install --no-cache-dir \
        git+https://github.com/aestrivex/bctpy@0.5.1 \
        graphviz \
        jupyterlab \
        mne \
        pandas \
        pydot \
        PyWavelets \
        sklearn \
 && apt-get remove --purge -y $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 tf \
 && useradd --create-home --shell /bin/bash --uid 1000 --gid tf tf

ARG NO_WANDER_DIR=/opt/no_wander
COPY app/src $NO_WANDER_DIR
ENV PYTHONPATH=$PYTHONPATH:$NO_WANDER_DIR/..

WORKDIR /home/tf
USER tf
