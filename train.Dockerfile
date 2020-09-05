FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

RUN DEV_PACKAGES=" \
        git \
 " \
 && apt-get update \
 && apt-get install -y \
        graphviz=2.40.1-2 \
 && apt-get install -y $DEV_PACKAGES \
 && pip install --no-cache-dir \
        autoreject==0.2.1 \
        bctpy==0.5.2 \
        graphviz==0.13.2 \
        jupyterlab==2.0.1 \
        keras-multi-head==0.22.0 \
        keras-pos-embd==0.11.0 \
        keras-position-wise-feed-forward==0.6.0 \
        mne==0.20.1 \
        pandas==1.0.3 \
        pydot==1.4.1 \
        PyWavelets==1.1.1 \
        scikit-learn==0.22.2.post1 \
 && apt-get remove --purge -y $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 tf \
 && useradd --create-home --shell /bin/bash --uid 1000 --gid tf tf

ARG NO_WANDER_DIR=/opt/no_wander
COPY no_wander $NO_WANDER_DIR/no_wander
ENV PYTHONPATH=$PYTHONPATH:$NO_WANDER_DIR

WORKDIR /home/tf
USER tf
