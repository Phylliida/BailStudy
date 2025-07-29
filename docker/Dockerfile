# ────────────────────────────────────────────────────────────────
# CUDA + Python base chosen to work on RunPod GPU nodes
# ----------------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

LABEL maintainer="Danielle-Ensign <phylliida.dev@gmail.com>" \
      org.opencontainers.image.source="https://github.com/Phylliida/BailStudy"

# ----------------------------------------------------------------
# 1. OS dependencies
# ----------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        curl ca-certificates git sudo bash tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------
# 2. Create an unprivileged user with password-less sudo
# ----------------------------------------------------------------
ARG USER=app
ARG UID=1000
RUN useradd -m -s /bin/bash -u ${UID} ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USER}
WORKDIR /home/${USER}

# ----------------------------------------------------------------
# 3. Install Astral’s ultra-fast package manager `uv`
#    (https://github.com/astral-sh/uv)
# ----------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --y

ENV PATH="/home/${USER}/.cargo/bin:${PATH}"

# ----------------------------------------------------------------
# 4. Install the BailStudy repo + any other Python deps
# ----------------------------------------------------------------
RUN uv pip install --upgrade pip && \
    uv pip install "https://github.com/Phylliida/BailStudy.git"

