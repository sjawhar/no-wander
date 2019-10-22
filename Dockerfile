FROM sjawhar/muselsl

USER root
RUN DEV_PACKAGES=" \
        python3-pip \
    " \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-opengl \
 && apt-get install -y $DEV_PACKAGES \
 && pip3 install \
        psychopy \
        pygame \
 && apt-get purge -y --auto-remove $DEV_PACKAGES \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache

RUN usermod -a -G audio muselsl

USER muselsl
