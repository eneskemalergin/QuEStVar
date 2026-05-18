from __future__ import annotations

import re
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
SRC_DIR = REPO_ROOT / "src" / "questvar"
OPT_IN_EXTERNAL_TEST_FILES = {"_r_reference.py", "test_r_references.py"}


def _iter_python_test_files() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.glob("*.py")
        if path.name not in {"__init__.py", "test_suite_audit.py"}
    )


def test_default_suite_files_avoid_network_and_absolute_paths() -> None:
    forbidden_network_patterns = [
        r"\brequests\b",
        r"\burllib\b",
        r"\burllib3\b",
        r"\baiohttp\b",
        r"\bsocket\b",
        r"\bhttp\.client\b",
        r"\bsubprocess\b",
        r"\bwget\b",
        r"\bcurl\b",
    ]
    forbidden_absolute_path_patterns = [
        r"/home/",
        r"/Users/",
        r"C:\\\\",
        r"file://",
    ]
    forbidden_external_repo_patterns = [
        r'(?<!_)real/',
        r'(?<!_)ref/',
    ]

    for path in _iter_python_test_files():
        if path.name in OPT_IN_EXTERNAL_TEST_FILES:
            continue
        text = path.read_text()
        for pattern in forbidden_network_patterns + forbidden_absolute_path_patterns + forbidden_external_repo_patterns:
            assert re.search(pattern, text) is None, f"Unexpected self-contained suite violation in {path.name}: {pattern}"


def test_r_reference_path_is_explicitly_opt_in_only() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text()
    r_reference_test_text = (TESTS_DIR / "test_r_references.py").read_text()

    assert 'addopts = ["-m", "not r_reference"]' in pyproject_text
    assert 'pytestmark = pytest.mark.r_reference' in r_reference_test_text


def test_source_user_facing_terminology_prefers_feature_over_protein() -> None:
    allowed_exact_strings = {
        "for candidate in (\"feature_id\", \"protein_id\"):",
        "protein_ids: list[str] | None = None,",
        "feature_ids if feature_ids is not None else protein_ids",
        "Parameters 'feature_ids' and 'protein_ids' are aliases. Pass only one.",
        '"""Standalone Antler\'s plot with optional feature annotations.',
        'Backward-compatible alias for ``feature_ids``.',
        'Ignored if ``feature_ids`` or ``protein_ids`` is given.',
        "def annotate_proteins(",
        "feature_ids=protein_ids,",
    }
    forbidden_patterns = [
        r"Per-protein",
        r"tested proteins",
        r"excluded proteins",
        r"top proteins",
        r"protein status",
        r"one per protein",
        r"n_proteins",
    ]

    for path in sorted(SRC_DIR.rglob("*.py")):
        text = path.read_text()
        for allowed in allowed_exact_strings:
            text = text.replace(allowed, "")
        for pattern in forbidden_patterns:
            assert re.search(pattern, text) is None, f"Unexpected protein terminology in {path.relative_to(REPO_ROOT)}: {pattern}"