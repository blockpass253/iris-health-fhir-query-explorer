"""Integration test runner for natural-language FHIR questions.

Runs a fixed set of (question, expected_root) pairs through the LLM pipeline
(extract + bind) without executing SQL or requiring a live IRIS connection.
Each case calls the real OpenAI API; debug/llm.md captures the last run's
full LLM transcript for inspection.

Pass criteria per case:
  - bound.root_resource == expected_root
  - bound.feasibility.can_answer is True
  - bound.clarifying_question is None
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from app.debug.dump import debug_dir, start_message
from app.runtime.binding import bind_plan
from app.runtime.extraction import extract_plan
from app.runtime.models import BoundPlan, QueryPlan
from app.schema.persistence.registry_store import DEFAULT_REGISTRY_PATH, load_registry

console = Console()

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str]] = [
    # Patient-root (1–5)
    ("show diabetic patients", "Patient"),
    ("show female diabetic patients over 65", "Patient"),
    ("show diabetic patients taking metformin", "Patient"),
    ("show patients with A1c above 9", "Patient"),
    ("show patients with recent encounters", "Patient"),
    # Condition-root (6–10)
    ("show diabetes conditions", "Condition"),
    ("show active conditions", "Condition"),
    ("show COPD conditions", "Condition"),
    ("show conditions recorded in the last year", "Condition"),
    ("show conditions associated with female patients", "Condition"),
    # Observation-root (11–15)
    ("show A1c observations", "Observation"),
    ("show glucose observations above 200", "Observation"),
    ("show abnormal cholesterol observations", "Observation"),
    ("show observations recorded in the last 90 days", "Observation"),
    ("show observations for diabetic patients", "Observation"),
    # Encounter-root (16–20)
    ("show recent encounters", "Encounter"),
    ("show encounters in the last 30 days", "Encounter"),
    ("show encounters for diabetic patients", "Encounter"),
    ("show encounters involving female patients", "Encounter"),
    ("show COPD-related encounters", "Encounter"),
    # MedicationRequest-root (21–25)
    ("show metformin prescriptions", "MedicationRequest"),
    ("show active medication requests", "MedicationRequest"),
    ("show medication requests from the last 6 months", "MedicationRequest"),
    ("show prescriptions for diabetic patients", "MedicationRequest"),
    ("show prescriptions for patients over 65", "MedicationRequest"),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class TestCaseResult:
    index: int  # 1-based
    question: str
    expected_root: str
    passed: bool
    actual_root: str | None = None
    can_answer: bool | None = None
    clarifying_question: str | None = None
    missing: list[str] = field(default_factory=list)
    plan: QueryPlan | None = None
    bound: BoundPlan | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def _run_single_case(
    index: int,
    question: str,
    expected_root: str,
    registry,
) -> TestCaseResult:
    start_message(f"[case {index}] {question[:60]}")
    try:
        plan = await extract_plan([{"role": "user", "content": question}])
        bound = await bind_plan(plan, registry)
    except Exception:
        return TestCaseResult(
            index=index,
            question=question,
            expected_root=expected_root,
            passed=False,
            error=traceback.format_exc(),
        )

    root_ok = bound.root_resource == expected_root
    feasible = bound.feasibility.can_answer
    no_clarify = bound.clarifying_question is None
    passed = root_ok and feasible and no_clarify

    return TestCaseResult(
        index=index,
        question=question,
        expected_root=expected_root,
        passed=passed,
        actual_root=bound.root_resource,
        can_answer=bound.feasibility.can_answer,
        clarifying_question=bound.clarifying_question,
        missing=list(bound.feasibility.missing),
        plan=plan,
        bound=bound,
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

_WHERE_TO_LOOK = """\
### Where to look
- Wrong `root_resource` → extraction prompt (`app/runtime/prompts/extraction`)
- `can_answer=False` + missing resources → schema registry (run `index-schema`)
- `can_answer=False` + missing concepts → `app/runtime/coding.py` SYNONYMS / _SYNONYMS
- `can_answer=False` + missing paths → binding prompt or schema inference
- `clarifying_question` set → extraction/binding prompt needs more specificity
"""


def _write_report(results: list[TestCaseResult], registry_path: Path) -> Path:
    failures = [r for r in results if not r.passed]
    n_pass = len(results) - len(failures)
    n_total = len(results)

    lines: list[str] = [
        "# Test Question Failures",
        f"_{datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**{n_pass}/{n_total} passed — {len(failures)} failed**",
        f"Registry: `{registry_path}`",
        "",
        "---",
        "",
    ]

    for r in failures:
        actual = r.actual_root or "(error)"
        lines += [
            f'## FAIL [{r.index}] "{r.question}"',
            f"expected root: **{r.expected_root}** | got: **{actual}**",
            "",
        ]

        if r.error:
            lines += [
                "### Error",
                "```",
                r.error.strip(),
                "```",
                "",
            ]
        else:
            lines += [
                f"can_answer: `{r.can_answer}`",
                f"clarifying_question: `{r.clarifying_question}`",
                f"missing: `{r.missing}`",
                "",
            ]

            if r.plan is not None:
                lines += [
                    "### Extracted Plan (QueryPlan)",
                    "```json",
                    r.plan.model_dump_json(indent=2),
                    "```",
                    "",
                ]

            if r.bound is not None:
                lines += [
                    "### Bound Plan (BoundPlan)",
                    "```json",
                    r.bound.model_dump_json(indent=2),
                    "```",
                    "",
                ]

        lines += [
            _WHERE_TO_LOOK,
            "Re-run to see full LLM transcript in `debug/llm.md`:",
            "```",
            f"uv run iris test-questions --case {r.index}",
            "```",
            "",
            "---",
            "",
        ]

    report_path = debug_dir() / "test_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_test_questions(case_index: int | None) -> None:
    """Run all test cases (or one by 1-based index) and report results."""
    if not DEFAULT_REGISTRY_PATH.exists():
        console.print(
            f"[red]No registry at {DEFAULT_REGISTRY_PATH}.[/] "
            "Run [b]uv run iris index-schema TEST1 --namespace FHIRSERVER[/] first."
        )
        return

    registry = load_registry(DEFAULT_REGISTRY_PATH)

    if case_index is not None:
        if not (1 <= case_index <= len(TEST_CASES)):
            console.print(f"[red]--case must be between 1 and {len(TEST_CASES)}.[/]")
            return
        cases = [(case_index, *TEST_CASES[case_index - 1])]
    else:
        cases = [(i + 1, q, r) for i, (q, r) in enumerate(TEST_CASES)]

    results: list[TestCaseResult] = []
    for idx, question, expected_root in cases:
        label = f"[dim][{idx:>2}/{len(TEST_CASES)}][/] {question}"
        console.print(label, end=" … ")
        result = await _run_single_case(idx, question, expected_root, registry)
        results.append(result)
        if result.passed:
            console.print("[green]PASS[/]")
        else:
            console.print("[red]FAIL[/]")

    _print_summary(results, len(TEST_CASES))

    failures = [r for r in results if not r.passed]
    if failures:
        _print_failures(failures)
        report_path = _write_report(results, DEFAULT_REGISTRY_PATH)
        console.print(f"\n[dim]Full report written to [b]{report_path}[/][/]")


def _print_summary(results: list[TestCaseResult], total: int) -> None:
    n_pass = sum(1 for r in results if r.passed)
    n_run = len(results)
    color = "green" if n_pass == n_run else "red"
    console.print(f"\n[{color}]{n_pass}/{n_run} passed[/]")

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("#", style="dim", width=3)
    table.add_column("Question", min_width=45)
    table.add_column("Expected", width=18)
    table.add_column("Got", width=18)
    table.add_column("Feasible", width=8)
    table.add_column("Result", width=6)

    for r in results:
        root_ok = r.actual_root == r.expected_root if r.actual_root else False
        got = r.actual_root or "[red](error)[/]"
        got_styled = got if root_ok else f"[red]{got}[/]"
        feasible = (
            "[green]yes[/]"
            if r.can_answer
            else ("[red]no[/]" if r.can_answer is False else "[dim]—[/]")
        )
        status = "[green]PASS[/]" if r.passed else "[red]FAIL[/]"
        table.add_row(
            str(r.index),
            r.question,
            r.expected_root,
            got_styled,
            feasible,
            status,
        )

    console.print(table)


def _print_failures(failures: list[TestCaseResult]) -> None:
    console.print("\n[b red]Failures[/]\n")
    for r in failures:
        console.print(f"[b][{r.index}] {r.question}[/]")
        console.print(f"  expected root : {r.expected_root}")
        console.print(f"  actual root   : {r.actual_root or '(error)'}")
        console.print(f"  can_answer    : {r.can_answer}")
        if r.clarifying_question:
            console.print(f"  clarifying    : {r.clarifying_question}")
        if r.missing:
            console.print(f"  missing       : {r.missing}")

        if r.error:
            console.print(f"  [red]exception:[/]\n{r.error}")
        else:
            if r.plan is not None:
                console.print("\n  [b]Extracted Plan[/]")
                console.print(r.plan.model_dump_json(indent=2))
            if r.bound is not None:
                console.print("\n  [b]Bound Plan[/]")
                console.print(r.bound.model_dump_json(indent=2))
        console.print()
