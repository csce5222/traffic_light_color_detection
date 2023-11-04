FROM ubuntu:latest as compilesys

WORKDIR "/traffic_light_color_detection"

COPY traffic_light_color_detection /traffic_light_color_detection/traffic_light_color_detection
COPY tests /traffic_light_color_detection/tests
COPY poetry.lock /traffic_light_color_detection/poetry.lock
COPY pyproject.toml /traffic_light_color_detection/pyproject.toml
COPY docker-entrypoint.sh /traffic_light_color_detection/docker-entrypoint.sh

RUN mv /var/lib/apt/lists /var/lib/apt/oldlist
RUN mkdir -p /var/lib/apt/lists/partial
RUN apt-get clean
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git


# Install pyenv
RUN git clone https://github.com/pyenv/pyenv.git .pyenv
ENV HOME  /traffic_light_color_detection
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv install 3.9.5
RUN pyenv global 3.9.5
RUN eval "$(/pyenv/bin/pyenv init -)"

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/traffic_light_color_detection/.local/bin:$PATH"
RUN poetry install

# Setup local testing
RUN apt install unzip
ENV TEST_DATA=tests/data/kaggle_dataset
RUN yes | unzip $TEST_DATA/archive.zip -d $TEST_DATA
ENV PYTHONPATH="/traffic_light_color_detection/tests"


ENTRYPOINT ["sh", "docker-entrypoint.sh"]