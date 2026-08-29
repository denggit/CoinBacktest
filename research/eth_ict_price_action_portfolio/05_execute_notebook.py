#!/usr/bin/env python
"""Minimal local executor for the repository notebook.

The environment does not ship Jupyter/nbclient.  This executor runs code cells
in one Python namespace, records text outputs/execution counts, saves figures
requested by cells, and fails immediately on any exception.
"""

from __future__ import annotations

import ast
import argparse
import contextlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def execute_cell(source: str, namespace: dict[str, object]) -> tuple[str, object | None]:
    tree = ast.parse(source, mode="exec")
    stdout = io.StringIO()
    value: object | None = None
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            if prefix.body:
                exec(compile(prefix, "<notebook>", "exec"), namespace)
            value = eval(compile(ast.Expression(tree.body[-1].value), "<notebook>", "eval"), namespace)
        else:
            exec(compile(tree, "<notebook>", "exec"), namespace)
    return stdout.getvalue(), value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "research_notebook.ipynb"
    output_path = Path(__file__).resolve().parent / "ict_pa_v1" / "results" / "research_notebook_execution.json"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    count = 0
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        count += 1
        source = "".join(cell.get("source", []))
        try:
            captured, value = execute_cell(source, namespace)
        except Exception as exc:
            cell["execution_count"] = count
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": [],
                }
            ]
            if args.write_results:
                output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            raise
        outputs: list[dict[str, object]] = []
        if captured:
            outputs.append({"output_type": "stream", "name": "stdout", "text": captured.splitlines(keepends=True)})
        if value is not None:
            if hasattr(value, "to_string"):
                rendered = value.to_string(index=False)  # type: ignore[attr-defined]
            else:
                rendered = repr(value)
            outputs.append(
                {
                    "output_type": "execute_result",
                    "execution_count": count,
                    "metadata": {},
                    "data": {"text/plain": rendered.splitlines(keepends=True)},
                }
            )
        cell["execution_count"] = count
        cell["outputs"] = outputs
    if args.write_results:
        output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"executed {count} code cells: {output_path}")
    else:
        print(f"validated {count} code cells in memory: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
