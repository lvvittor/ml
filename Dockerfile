FROM python:3.10-slim-buster

# https://stackoverflow.com/a/74566142
RUN apt-get update && apt install -y chromium-driver 

ENV PATH="/root/.local/bin:$PATH"

RUN pip install pipx
RUN pipx install poetry
ENV PATH="/root/.local/pipx/venvs/poetry/bin/:$PATH"

COPY pyproject.toml /project/pyproject.toml
COPY poetry.lock /project/poetry.lock

WORKDIR /project

RUN --mount=type=cache,target=/root/.cache/pip \
    poetry install --with dev --no-root

COPY . /project

RUN --mount=type=cache,target=/root/.cache/pip \
    poetry install --with dev --no-interaction