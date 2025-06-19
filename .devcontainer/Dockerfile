FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the default user to vscode
USER vscode

# Set working directory
WORKDIR /workspaces/PYEDGIE
