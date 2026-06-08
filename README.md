# iris-search-agent

`iris-search-agent` is a transparent natural-language query agent for
InterSystems IRIS for Health FHIR SQL Builder projections.

It lets a user ask clinical questions such as:

- "Show diabetic patients with recent encounters"
- "Count patients with A1c above 9 in the last 6 months"
- "Top 5 medications prescribed in the last 6 months"

The system does not hide the query plan or SQL. It shows the extracted intent,
the grounded schema bindings, the generated SQL, and the final results so the
user can see exactly how the answer was produced.

## What It Does

The project currently supports an end-to-end runtime pipeline:

1. Index a FHIR SQL Builder projection into a semantic registry.
2. Extract a structured clinical query plan from natural language.
3. Ground that plan against the indexed schema.
4. Generate deterministic IRIS SQL from the grounded plan.
5. Execute the SQL and render results.
6. Ask for clarification or suggest projection extensions when the schema cannot
   fully answer the question.

There are two interfaces over the same runtime:

- A Textual TUI with multi-turn conversation memory and clarification handling.
- A headless CLI used for developer testing and debugging.

## Design Principles

- SQL is always shown to the user.
- LLM output is schema-grounded and deterministically validated.
- The runtime prefers explainability and demo reliability over broad SQL
  expressiveness.
- Schema indexing and semantic inference stay separate from runtime query
  execution.
- Unsupported queries fail explicitly instead of silently guessing.

## Current Query Shape

The current engine is strongest at:

- Patient cohort discovery
- Count queries
- Root-resource ranked aggregates such as "top medications" or "most common
  encounter statuses"
- Relative time windows such as "last 6 months" or "last year"
- Concept-driven filters such as diabetes, metformin, or A1c
- Multi-turn refinements in the TUI

The current SQL strategy is deliberately narrow:

- The root resource anchors the query.
- Root filters apply directly on the root table.
- Non-root resources are correlated through `EXISTS`, typically via patient
  identity.
- Ranked queries group the root resource and order by count.

Current limitations:

- No general joins for projecting related-resource columns.
- No sorting by related-resource columns.
- No arbitrary column projection from correlated resources.
- Query support is intentionally narrower than full SQL or full FHIR search.

## Architecture

### Indexing

The indexing pipeline introspects a FHIR SQL Builder projection, parses FHIR
paths from column metadata, infers physical and semantic relationships, and
persists a semantic registry to `data/schema_registry.json`.

No LLM is used in this stage.

#### Coding Dictionary

During `index-schema`, the pipeline also samples each coding child table's
`system`, `code`, and `display` columns to discover terminology codes present
in the database. The results are merged with a set of hardcoded baseline entries
(hardcoded entries always take precedence) and written to
`data/coding_dictionary.json`.

At runtime, `lookup_codes()` reads from this file when it exists, falling back
to the hardcoded entries when the file is absent (e.g. in test environments).

Re-run `index-schema` after loading new patient data to refresh the coding
dictionary.

### Runtime

The runtime query flow is:

`extract -> bind -> (clarify | suggest projection -> clarify | run SQL)`

- `extract`: LLM turns conversation history into a typed `QueryPlan`.
- `bind`: LLM proposes schema mappings and deterministic validation turns them
  into a `BoundPlan`.
- `run SQL`: deterministic SQL generation and execution over IRIS.
- `clarify`: the TUI can pause and resume when the schema-aware stage needs user
  input.
- `suggest projection`: when the schema is missing required resources or fields,
  the runtime suggests what to add to the FHIR projection and re-index.

The CLI and TUI share the same orchestrator and query graph, but the TUI is the
intended demo interface.

## Stack

- **Python 3.12**, managed with [uv](https://docs.astral.sh/uv/)
- `intersystems-irispython` for IRIS DB-API access
- OpenAI SDK for structured extraction and binding
- LangGraph for multi-turn query orchestration
- Pydantic v2 for typed query-plan models
- Typer + Textual + Rich for the CLI and TUI
- structlog for indexing/runtime logs

## Usage

### 1. Index a schema

Introspect a FHIR SQL Builder projection and persist the semantic registry:

```bash
uv run iris index-schema TEST1 --namespace FHIRSERVER
uv run iris tui
```

In the TUI, run:

```text
/index-schema TEST1 --namespace FHIRSERVER
```

`--namespace` overrides `IRIS_NAMESPACE` for that run. This is useful when FHIR
projections live in a namespace such as `FHIRSERVER` instead of `USER`.

### 2. Use the interactive TUI

```bash
uv run iris-agent tui
make tui
```

In the TUI:

- lines starting with `/` are commands
- all other input is treated as a natural-language clinical query
- follow-up turns can refine the previous question
- clarification prompts pause the graph and resume on reply

Example flow:

```text
Show diabetic patients
just the ones over 65
count them
```

### 3. Developer CLI

The CLI exists mainly for developer testing, debugging, and inspecting the
runtime stages outside the TUI.

```bash
uv run iris query "Show diabetic patients with recent encounters"
uv run iris query "Count patients with A1c above 9 in the last 6 months"
uv run iris query "Top 5 medications prescribed in the last 6 months"
```

The packaged console entrypoints `iris` and `iris-agent` both point to the same
CLI app.

For answerable questions, the CLI prints:

- the extracted plan
- the grounded plan
- the generated SQL
- the results

For infeasible questions, it prints:

- the extracted plan
- the partial grounded plan
- missing capability/schema details
- suggested FHIR resources or fields to project when available

## Setup

Prerequisites:

- [uv](https://docs.astral.sh/uv/)
- Python 3.12
- A FHIR SQL Builder projection to index
- `OPENAI_API_KEY` for runtime extraction and binding

You can connect either to your own IRIS instance or to the demo container
defined in [compose.yaml](compose.yaml).

### Local Python Setup

Install dependencies and initialize local tooling:

```bash
make install
cp .env.example .env
make run
```

`make run` is a connectivity smoke test that executes `SELECT $ZVERSION`.

### Docker Demo IRIS

The repository includes a Docker Compose service for a prebuilt IRIS demo
image:

```bash
cp .env.example .env
make iris-up
make run
```

Useful Docker targets:

- `make iris-up`: start the IRIS demo container in the background
- `make iris-down`: stop and remove the Compose services
- `make iris-logs`: tail the IRIS container logs

The Compose service publishes these ports by default:

- `1972` via `IRIS_PORT` for IRIS superserver access
- `52773` via `IRIS_WEB_PORT` for the IRIS web apps

### Connection Settings

Configure IRIS access through environment variables or `.env`:

| Variable         | Default     | Description             |
| ---------------- | ----------- | ----------------------- |
| `IRIS_HOST`      | `localhost` | IRIS hostname           |
| `IRIS_PORT`      | `1972`      | Superserver port        |
| `IRIS_NAMESPACE` | `FHIRSERVER` | Namespace to connect to |
| `IRIS_USERNAME`  | `_SYSTEM`   | Username                |
| `IRIS_PASSWORD`  | `SYS`       | Password                |

Additional Docker-only port overrides:

| Variable           | Default | Description                        |
| ------------------ | ------- | ---------------------------------- |
| `IRIS_WEB_PORT`    | `52773` | Published IRIS web application port  |

Configure the LLM runtime:

| Variable         | Default        | Description                            |
| ---------------- | -------------- | -------------------------------------- |
| `OPENAI_API_KEY` | _(unset)_      | OpenAI API key                         |
| `OPENAI_MODEL`   | `gpt-5.4-nano` | Model used for structured plan parsing |

## Repository Map

- [config.py](config.py): `IrisSettings` and env-driven configuration
- [iris_client.py](iris_client.py): thin IRIS DB-API wrapper
- [main.py](main.py): connectivity smoke test
- [compose.yaml](compose.yaml): Docker Compose service for the demo IRIS image
- [app/commands/](app/commands): CLI/TUI orchestration and rendering
- [app/runtime/](app/runtime): extraction, binding, diagnosis, graph, and SQL
  generation
- [app/schema/](app/schema): indexing, parsing, graph building, and registry
  persistence
- [app/tui/](app/tui): interactive Textual interface
- [data/schema_registry.json](data/schema_registry.json): persisted semantic
  registry output
- [data/coding_dictionary.json](data/coding_dictionary.json): generated coding
  concept-to-code dictionary (produced by `index-schema`)
- [scripts/](scripts): Docker snapshot and GHCR publishing helpers

## Development

Common tasks are wrapped in the `Makefile`:

| Command          | Description                                    |
| ---------------- | ---------------------------------------------- |
| `make install`   | Sync dependencies and install the git hook     |
| `make run`       | Run the IRIS connectivity smoke test           |
| `make tui`       | Launch the interactive Textual TUI             |
| `make lint`      | Lint with ruff                                 |
| `make format`    | Format with ruff                               |
| `make typecheck` | Type-check with pyright                        |
| `make test`      | Run the test suite                             |
| `make check`     | Run lint, typecheck, and tests                 |
| `make precommit` | Run all pre-commit hooks across the repo       |
| `make clean`     | Remove caches and build artifacts              |

Run `make help` to list targets.

The current automated test suite covers binding, FHIR path parsing, grounding,
graph behavior, query formatting, and SQL generation.
