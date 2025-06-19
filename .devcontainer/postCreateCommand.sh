#! /usr/bin/env bash

set -e

echo "Setting up PYEDGIE development environment..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Dependencies
echo "Installing dependencies with uv..."
uv sync

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
uv run pre-commit install --install-hooks

echo "Development environment setup complete!"
