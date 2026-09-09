###############################################################################
# POETRY BASE IMAGE - System packages + poetry env vars
###############################################################################
FROM python:3.10-slim-bookworm AS python-poetry-base
ARG POETRY_VERSION=""

ENV POETRY_VERSION=${POETRY_VERSION}
ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1
ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update && \
    apt-get install --no-install-recommends -y alsa-utils v4l-utils libglib2.0-0 libexpat1 libx11-6 libxrender1 libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


###############################################################################
# POETRY BUILDER - Installs Poetry via curl
###############################################################################
FROM python-poetry-base AS python-poetry-builder
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://install.python-poetry.org | python3 -


###############################################################################
# POETRY RUNTIME - Clean image with poetry binary
###############################################################################
FROM python-poetry-base AS python-poetry
COPY --from=python-poetry-builder $POETRY_HOME $POETRY_HOME


###############################################################################
# OBFUSCATED - PyArmor code obfuscation
###############################################################################
FROM python:3.10-slim-bookworm AS obfuscated
RUN pip install pyarmor==8.5.11
WORKDIR /app
ADD . /app
# Pyarmor 原始碼加密授權文件 - 購買資訊  https://pyarmor.readthedocs.io/zh/latest/licenses.html
# RUN pyarmor reg pyarmor-regfile-5492.zip && \
#    pyarmor gen --output ./ --recursive ./ && \
#    rm pyarmor-regfile-5492.zip


###############################################################################
# DEPENDENCIES BUILDER - Install all Python dependencies globally
###############################################################################
FROM python-poetry AS dependencies-builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build
ADD pyproject.toml poetry.lock /build/
ADD libs/ /build/libs/
ADD ./libs/librknnrt.so /usr/lib/librknnrt.so

# Fix wheel hashes in poetry.lock in case wheels were rebuilt
RUN for whl in libs/m01_frame_processor-*.whl; do \
        hash=$(python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$whl"); \
        fname=$(basename "$whl"); \
        sed -i "s|${fname}\", hash = \"sha256:[a-f0-9]*\"|${fname}\", hash = \"sha256:${hash}\"|" poetry.lock 2>/dev/null || true; \
    done

RUN poetry lock && \
    poetry install --no-cache && \
    pip install libs/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl && \
    poetry cache clear --all -n . && \
    rm -rf /root/.cache/pypoetry/{cache,artifacts} /root/.cache/pip

# Use obfuscated application code
WORKDIR /app
COPY --from=obfuscated /app /app
RUN rm -rf /app/libs


###############################################################################
# RUNTIME - Final minimal image
###############################################################################
FROM python-poetry-base AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy globally installed Python packages and executables
COPY --from=dependencies-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=dependencies-builder /usr/local/bin /usr/local/bin

# Copy RKNN runtime shared library
COPY --from=dependencies-builder /usr/lib/librknnrt.so /usr/lib/librknnrt.so

# Copy obfuscated application code
COPY --from=dependencies-builder /app /app

# Setup ALSA audio state
RUN mkdir -p /var/lib/alsa
COPY ./asound.state /var/lib/alsa/asound.state

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
