FROM tensorflow/tensorflow:2.1.0-gpu-py3-jupyter

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
        keras \
        mne \
        pandas \
        pydot \
        PyWavelets \
        sklearn \
 && apt-get remove --purge -y $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 tf \
 && useradd --create-home --shell /bin/bash --uid 1000 --gid tf tf

WORKDIR /home/tf
COPY app/src ./no_wander
USER tf
