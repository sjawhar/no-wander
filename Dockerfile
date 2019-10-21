FROM sjawhar/muselsl

USER root
RUN DEV_PACKAGES="\
        python3-pip \
    " \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3-opengl \
 && apt-get install -y $DEV_PACKAGES \
 && pip3 install \
        psychopy \
 && apt-get remove -y $DEV_PACKAGES \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache

USER muselsl
