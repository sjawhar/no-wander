FROM sjawhar/pipenv:3.7-2020.6.2 as packages
USER root
WORKDIR /scratch
COPY app/Pipfile app/Pipfile.lock ./
RUN PIP_TARGET=/scratch/packages PIP_IGNORE_INSTALLED=1 pipenv install --system

FROM sjawhar/muselsl
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        freeglut3=2.8.1-3 \
        libglu1-mesa=9.0.0-2.1+b3 \
 && rm -rf /var/lib/apt/lists/*
RUN usermod -a -G audio muselsl

ARG PACKAGES_DIR=/opt/site-packages
COPY --from=packages /scratch/packages $PACKAGES_DIR
ENV PYTHONPATH=$PYTHONPATH:$PACKAGES_DIR

ARG NO_WANDER_DIR=/opt/no_wander
COPY app/src $NO_WANDER_DIR
ENV PYTHONPATH=$PYTHONPATH:$NO_WANDER_DIR/..

WORKDIR /home/muselsl
USER muselsl
