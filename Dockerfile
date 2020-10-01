FROM sjawhar/pipenv:3.7-2020.8.13 as packages
USER root
WORKDIR /scratch
COPY Pipfile Pipfile.lock ./

ARG PACKAGES_DIR=/scratch/packages
RUN PIP_PREFIX=${PACKAGES_DIR} \
    PIP_IGNORE_INSTALLED=1 \
    pipenv install --system --ignore-pipfile --deploy

ARG LIBLSL_VERSION=1.13.1
RUN apt-get update \
 && apt-get install -y \
        wget \
 && wget https://github.com/sccn/liblsl/releases/download/${LIBLSL_VERSION}/liblsl-${LIBLSL_VERSION}-manylinux2010_x64.so \
        -O ${PACKAGES_DIR}/lib/python3.7/site-packages/pylsl/liblsl64.so

FROM sjawhar/muselsl as base
USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        freeglut3=2.8.1-3 \
        libglu1-mesa=9.0.0-2.1+b3 \
 && rm -rf /var/lib/apt/lists/*
RUN usermod -a -G audio muselsl

ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib/python3.7/site-packages

FROM base as prod
ARG PACKAGES_DIR=/scratch/packages
COPY --from=packages ${PACKAGES_DIR} /usr/local

ARG NO_WANDER_DIR=/opt/no_wander
COPY no_wander ${NO_WANDER_DIR}/no_wander
ENV PYTHONPATH=${PYTHONPATH}:${NO_WANDER_DIR}

WORKDIR /home/muselsl
USER muselsl

FROM base as dev
USER root
RUN apt-get update \
 && apt-get install -y \
        git \
        python3-pip \
 && pip3 install pipenv==2020.8.13

WORKDIR /app
RUN chown muselsl:muselsl .

ARG PACKAGES_DIR=/scratch/packages
COPY --from=packages ${PACKAGES_DIR} /usr/local
COPY --chown=muselsl:muselsl Pipfile Pipfile.lock ./
RUN pipenv install --system --ignore-pipfile --deploy --dev

COPY --chown=muselsl:muselsl . ./
RUN pip3 install -e .

USER muselsl
ENV SHELL /bin/bash
ENTRYPOINT []
