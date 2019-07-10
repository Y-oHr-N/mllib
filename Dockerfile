FROM python:3.7

ENV MLLIB $HOME/mllib

WORKDIR $MLLIB

COPY . $MLLIB

RUN pip install notebook
RUN pip install .
