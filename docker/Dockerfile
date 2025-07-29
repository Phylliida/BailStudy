# ────────────────────────────────────────────────────────────────
# CUDA + Python base chosen to work on RunPod GPU nodes
# ----------------------------------------------------------------
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL maintainer="Danielle-Ensign <phylliida.dev@gmail.com>" \
      org.opencontainers.image.source="https://github.com/Phylliida/BailStudy"

# modified from https://github.com/jplhughes/dotfiles/blob/master/runpod/johnh_dev.Dockerfile
# Install SSH and other dependencies
RUN apt-get update && \
    apt-get install -y openssh-server sudo less nano htop ncdu && \
    mkdir /var/run/sshd

# Create a directory for SSH keys
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Update SSH configuration
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# Add sudo stuff
RUN su -c 'apt-get update && apt-get install -y sudo'
RUN sudo apt-get install -y less nano htop ncdu nvtop lsof rsync btop jq

# Setup virtual environment
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup python with uv
RUN source $HOME/.local/bin/env && \
    uv python install 3.11 && \
    uv venv

RUN source $HOME/.local/bin/env && \
    uv pip install --no-cache-dir ipykernel huggingface_hub https://github.com/Phylliida/BailStudy.git


