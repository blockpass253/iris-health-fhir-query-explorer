# iris-search-agent

### Design principles

- SQL is LLM-generated, but always shown to the user — transparency over a hidden black box.
- LLM proposals are grounded against the indexed schema: extraction and binding are
  deterministically validated so the model selects real tables/paths/codes rather than inventing them.
- No autonomous agents.
- Prioritize explainability, transparency, and demo reliability over breadth.
- Keep semantic interpretation and schema grounding cleanly separated from SQL generation.

## Stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- `intersystems-irispython` — official IRIS DB-API driver
- OpenAI SDK — structured semantic extraction (resource selection)
- Pydantic v2 — typed query-plan models
- Typer + Textual + Rich — CLI and interactive analytics TUI
- structlog — structured logging for the indexing pipeline

## Current state

The IRIS connectivity layer and the deterministic **schema indexing pipeline**
are in place. The pipeline introspects a FHIR SQL Builder projection, parses the
FHIR paths embedded in column descriptions, infers physical and semantic
relationships, builds a semantic graph, and persists a JSON registry — with no
LLM involvement.

The first runtime stage, **LLM-assisted semantic resource selection**, is also
built: a natural-language question is turned into a compact context derived from
the indexed registry, the LLM selects the relevant root resources (structured,
validated output — it cannot invent resources), and the semantic graph is
deterministically narrowed to those resources plus any bridge resources needed
to connect them. The query-plan → SQL-generation layers are not yet built.

- [config.py](config.py) — `IrisSettings` (pydantic-settings), env-driven with an `IRIS_` prefix.
- [iris_client.py](iris_client.py) — thin DB-API wrapper; `run_query()` returns list-of-dict rows for result sets, else `rowcount`.
- [main.py](main.py) — smoke test that runs `SELECT $ZVERSION`.
- [app/](app/) — the indexing pipeline (introspection, FHIR-path parsing, semantic inference, graph, persistence), the runtime resource-selection layer ([app/runtime/](app/runtime/), [app/llm/](app/llm/)), and the Typer CLI and Textual TUI.

## Indexing a schema

Introspect a FHIR SQL Builder projection and write the semantic registry to
`data/schema_registry.json`:

```bash
uv run iris index-schema TEST1 --namespace FHIRSERVER   # or python -m app.cli ...
uv run iris tui                                          # interactive TUI; then: /index-schema TEST1 --namespace FHIRSERVER
```

`--namespace` overrides `IRIS_NAMESPACE` for the run (FHIR projections often live
in a dedicated namespace such as `FHIRSERVER`, separate from the default `USER`).
The pipeline reports counts of tables, columns, physical relationships, and
semantic FHIR relationships, and renders the inferred resource graph. Every
inferred relationship carries a `confidence` and a `rationale` for
explainability. `Base`/infrastructure tables and system columns are excluded.

## Asking questions

Once a schema has been indexed (the registry at `data/schema_registry.json` is
the runtime source of truth), ask clinical questions in plain English. This runs
LLM resource selection and renders the narrowed semantic subgraph; it does **not**
yet generate or execute SQL. Requires `OPENAI_API_KEY` (see below); does **not**
require a live IRIS connection, since it reads the persisted registry.

```bash
uv run iris query "Show diabetic patients with recent encounters"   # headless
uv run iris tui                                                      # then type the question (no leading slash)
make tui                                                             # shortcut for the TUI
```

In the TUI, lines starting with `/` are commands (e.g. `/index-schema`); any
other line is treated as a natural-language question. Selected resources, the
relevant relationships, and the model's reasoning are printed. Use
`iris query --registry <path>` to point at a non-default registry file.

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

Resource selection additionally needs OpenAI credentials (same `.env` file):

| Variable         | Default        | Description                          |
| ---------------- | -------------- | ------------------------------------ |
| `OPENAI_API_KEY` | _(unset)_      | OpenAI API key for resource selection |
| `OPENAI_MODEL`   | `gpt-5.4-nano` | Model used for structured extraction |

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
