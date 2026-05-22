# Contributing

## Development setup

```bash
git clone https://github.com/eneskemalergin/QuEStVar
cd QuEStVar
uv sync --dev
```

## Running tests

```bash
uv run pytest
```

## Linting and type checking

```bash
uv run ruff check src/questvar/ tests/
uv run ruff format --check src/questvar/ tests/
uv run mypy src/questvar/
```

## Building documentation

```bash
# Convert tutorial scripts to notebooks
uv run jupytext --to notebook docs/notebooks/*.py

# Build the site
uv run mkdocs build

# Serve locally
uv run mkdocs serve
```

## Pull request workflow

1. Create a feature branch from `main`
2. Make your changes
3. Run tests, linting, and type checking
4. Update or add tests as needed
5. Build the docs site and verify no warnings
6. Open a pull request
