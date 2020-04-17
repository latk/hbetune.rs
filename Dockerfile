FROM rust:1.42

RUN apt-get update \
  && apt-get install -y cmake gfortran \
  && apt-get clean

WORKDIR /usr/src/hbetune
COPY . .
RUN make test-build  # separate so that it can be cached before test failure
RUN make test install

CMD ["hbetune"]
