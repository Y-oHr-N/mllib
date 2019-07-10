FROM python:3.7

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
ENV PATH ${HOME}/.local/bin:${PATH}

USER root

# Create a user with a home directory
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
COPY . ${HOME}
RUN chown -R ${NB_UID} ${HOME}

USER ${USER}

WORKDIR ${HOME}

# Install the packages
RUN pip install --user --quiet notebook
RUN pip install --user --quiet .[develop]
