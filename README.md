# QuantMinds 2025

## Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Installing uv

If you don't have uv installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installing Dependencies

Install project dependencies using uv:

```bash
# Install a package
uv sync --all-extras --all-packages
```

### Running Python Scripts

Use `uv run` to execute Python scripts with the managed environment:

```bash
source .venv/bin/activate
```
