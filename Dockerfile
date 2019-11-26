FROM sjawhar/muselsl

USER root
COPY app/Pipfile app/Pipfile.lock ./
RUN DEV_PACKAGES=" \
        pipenv \
    " \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-opengl \
 && apt-get install -y $DEV_PACKAGES \
 && pipenv install --system \
 && apt-get purge -y --auto-remove $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache \
 && rm ./Pipfile*

RUN usermod -a -G audio muselsl

USER muselsl
