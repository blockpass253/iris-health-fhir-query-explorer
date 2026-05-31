# iris-search-agent

### Design principles

- No arbitrary LLM-generated SQL — SQL generation stays deterministic and inspectable.
- No autonomous agents.
- Prioritize explainability, transparency, and demo reliability over breadth.
- Keep semantic interpretation and SQL generation cleanly separated.

## Stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- `intersystems-irispython` — official IRIS DB-API driver
- OpenAI SDK — structured semantic extraction
- Pydantic v2 — typed query-plan models
- SQLGlot — deterministic SQL building *(planned)*
- Textual + Typer + Rich — interactive analytics TUI *(planned)*

## Current state

Early scaffold — only the IRIS connectivity layer exists; the
semantic/mapping/SQL-generation pipeline and TUI are not yet built.

- [config.py](config.py) — `IrisSettings` (pydantic-settings), env-driven with an `IRIS_` prefix.
- [iris_client.py](iris_client.py) — thin DB-API wrapper; `run_query()` returns list-of-dict rows for result sets, else `rowcount`.
- [main.py](main.py) — smoke test that runs `SELECT $ZVERSION`.

## Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/), Python 3.12, and a running
**InterSystems IRIS for Health** instance.

```bash
make install              # uv sync + install the pre-commit git hook
cp .env.example .env      # then fill in your instance's connection details
make run                  # run the IRIS connectivity smoke test
```

### Connection settings

Configure via environment variables or a local `.env` file (all keys use the
`IRIS_` prefix):

| Variable         | Default     | Description             |
| ---------------- | ----------- | ----------------------- |
| `IRIS_HOST`      | `localhost` | IRIS hostname           |
| `IRIS_PORT`      | `1972`      | Superserver port        |
| `IRIS_NAMESPACE` | `USER`      | Namespace to connect to |
| `IRIS_USERNAME`  | `_SYSTEM`   | Username                |
| `IRIS_PASSWORD`  | `SYS`       | Password                |

## Development

Common tasks are wrapped in the `Makefile` (all run via `uv run`):

| Command          | Description                                    |
| ---------------- | ---------------------------------------------- |
| `make install`   | Sync dependencies and install the git hook     |
| `make run`       | Run the IRIS connectivity smoke test           |
| `make lint`      | Lint with ruff                                 |
| `make format`    | Format with ruff                               |
| `make typecheck` | Type-check with pyright                        |
| `make test`      | Run the test suite                             |
| `make check`     | Run lint, typecheck, and tests (local CI gate) |
| `make precommit` | Run all pre-commit hooks across the repo       |
| `make clean`     | Remove caches and build artifacts              |

Run `make help` to list available targets. Pre-commit hooks (ruff lint +
format, basic file checks, and pyright) run automatically on `git commit` once
`make install` has been run.
