"""Export pytest performance JSON artifacts to CSV or Markdown; stdlib only.

Each row carries the required elements -- operator name, case name, e2e mean
and profiler span mean -- plus the context needed to read them (source file,
target, implementation, workload parameters). Unlike compare_perf.py this is
a tolerant analysis view: incomplete runs export what they contain and report
what was skipped or missing on stderr.

Usage:
    python scripts/export_perf.py current.json                       # MD to stdout
    python scripts/export_perf.py npu.json cuda.json -o compare.csv  # merged CSV
    python scripts/export_perf.py current.json -o report.md          # MD file
"""

import argparse
import csv
import json
import statistics
import sys

from pathlib import Path

COLUMNS = ("source", "target", "op", "case", "implementation", "parameters",
           "e2e_mean_us", "span_mean_us")
PARAMS_HIDDEN = ("op", "implementation", "workload_sha256", "phase")


def mean_us(metric):
    """Arithmetic mean over raw samples; fall back to the median summary."""
    if not isinstance(metric, dict):
        return None
    samples = metric.get("samples")
    if samples:
        return statistics.fmean(float(value) for value in samples)
    median = metric.get("median")
    return float(median) if median is not None else None


def span_metric(metrics):
    """The profiler span metric for this report, whatever its reduction."""
    for name in ("profiler_span_us", "profiler_sum_us"):
        if name in metrics:
            return metrics[name]
    return None


def case_name(node_id):
    """The parametrized part of a pytest id, or the test name if unparametrized."""
    if "::" in node_id:
        node_id = node_id.split("::", 1)[1]
    if "[" in node_id and node_id.endswith("]"):
        return node_id[node_id.index("[") + 1:-1]
    return node_id


def report_rows(path):
    with Path(path).open(encoding="utf-8") as report_file:
        report = json.load(report_file)

    environment = report.get("environment") or {}
    target = environment.get("target", "")
    source = Path(path).name
    if report.get("exitstatus") not in (0, None):
        print(f"[export_perf] warning: {source} exitstatus={report['exitstatus']}", file=sys.stderr)
    if report.get("skipped"):
        print(f"[export_perf] warning: {source} skipped {len(report['skipped'])} case(s)", file=sys.stderr)

    rows = []
    for case in report.get("cases", []):
        parameters = case.get("parameters") or {}
        metrics = case.get("metrics") or {}
        shown = [
            f"{key}={value}"
            for key, value in parameters.items()
            if key not in PARAMS_HIDDEN
        ]
        e2e = mean_us(metrics.get("e2e_us"))
        span = mean_us(span_metric(metrics))
        rows.append({
            "source": source,
            "target": target,
            "op": parameters.get("op", ""),
            "case": case_name(case.get("id", "")),
            "implementation": parameters.get("implementation", ""),
            "parameters": ",".join(shown),
            "e2e_mean_us": "" if e2e is None else f"{e2e:.3f}",
            "span_mean_us": "" if span is None else f"{span:.3f}",
        })
    return rows


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def to_markdown(rows):
    header = "| " + " | ".join(COLUMNS) + " |"
    align = "| " + " | ".join("---:" if name.endswith("_us") else "---" for name in COLUMNS) + " |"
    lines = [header, align]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row[name]) for name in COLUMNS) + " |")
    lines.append("")
    lines.append(f"{len(rows)} case(s) exported.")
    return "\n".join(lines)


def to_csv(rows, stream):
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+", help="performance result JSON file(s)")
    parser.add_argument("-o", "--output", default=None,
                        help="output file; format follows the suffix (.csv/.md), default MD on stdout")
    parser.add_argument("--format", choices=("csv", "md"), default=None,
                        help="override the output format")
    args = parser.parse_args(argv)

    rows = []
    for path in args.reports:
        try:
            rows.extend(report_rows(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"{path}: {error}")
    if not rows:
        parser.error("no performance cases found in the given report(s)")

    fmt = args.format
    if fmt is None:
        suffix = Path(args.output).suffix.lower() if args.output else ".md"
        fmt = "csv" if suffix == ".csv" else "md"

    if args.output is None:
        if fmt == "csv":
            to_csv(rows, sys.stdout)
        else:
            print(to_markdown(rows))
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        with output.open("w", encoding="utf-8", newline="") as stream:
            to_csv(rows, stream)
    else:
        output.write_text(to_markdown(rows) + "\n", encoding="utf-8")
    print(f"[export_perf] wrote {len(rows)} row(s) to {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
