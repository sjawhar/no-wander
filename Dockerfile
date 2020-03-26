FROM sjawhar/muselsl

USER root
COPY app/Pipfile app/Pipfile.lock ./
RUN DEV_PACKAGES=" \
        python3-pip \
    " \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-opengl=3.1.0+dfsg-2 \
 && apt-get install -y $DEV_PACKAGES \
 && pip3 install pipenv \
 && pipenv install --system \
 && pip3 uninstall -y pipenv \
 && apt-get purge -y --auto-remove $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache

RUN usermod -a -G audio muselsl

ARG NO_WANDER_DIR=/opt/no_wander
COPY app/src $NO_WANDER_DIR
ENV PYTHONPATH=$PYTHONPATH:$NO_WANDER_DIR/..

WORKDIR /home/muselsl
USER muselsl
