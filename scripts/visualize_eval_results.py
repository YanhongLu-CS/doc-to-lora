#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path


DEFAULT_METRICS = [
    "qa_f1_score",
    "qa_precision",
    "qa_recall",
    "runtime",
    "samples_per_second",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Doc-to-LoRA evaluation results from one or more run directories."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run spec in the form LABEL=DIR. "
            "Example: batch=/path/to/eval-results-80000/run1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the merged summary and HTML report.",
    )
    parser.add_argument(
        "--csv-name",
        default="evaluation_results_generation.csv",
        help="Result CSV filename inside each run directory.",
    )
    parser.add_argument(
        "--group-len",
        default="overall",
        help="Which group_len row to visualize. Default: overall",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metrics to visualize.",
    )
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run value: {spec!r}. Expected LABEL=DIR.")
    label, directory = spec.split("=", 1)
    label = label.strip()
    directory = Path(directory.strip()).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid --run value: {spec!r}. Empty label.")
    return label, directory


def coerce_number(value: str):
    if value in {"", "N/A", "None", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_run_rows(label: str, run_dir: Path, csv_name: str, group_len: str):
    csv_path = run_dir / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find CSV: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("group_len") != group_len:
                continue
            row["run_label"] = label
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def collect_results(run_specs, csv_name: str, group_len: str):
    all_rows = []
    for label, run_dir in run_specs:
        all_rows.extend(load_run_rows(label, run_dir, csv_name, group_len))
    return all_rows


def write_summary_csv(rows, output_path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(rows, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def unique_in_order(values):
    out = []
    seen = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def svg_grouped_bar_chart(title: str, task_names, run_labels, values_by_run_and_task):
    width = 980
    height = 340
    margin_left = 90
    margin_right = 30
    margin_top = 45
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    valid_values = [
        value
        for run_values in values_by_run_and_task.values()
        for value in run_values.values()
        if value is not None
    ]
    max_value = max(valid_values) if valid_values else 1.0
    if max_value <= 0:
        max_value = 1.0

    colors = [
        "#3366cc",
        "#dc3912",
        "#ff9900",
        "#109618",
        "#990099",
        "#0099c6",
    ]

    svg = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#222}.small{font-size:11px}.title{font-size:16px;font-weight:700}.axis{stroke:#666;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="24" text-anchor="middle">{html.escape(title)}</text>',
    ]

    # grid + y labels
    for i in range(6):
        y_val = max_value * i / 5
        y = margin_top + plot_height - (plot_height * i / 5)
        svg.append(
            f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" />'
        )
        svg.append(
            f'<text class="small" x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end">{y_val:.3f}</text>'
        )

    svg.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" />'
    )
    svg.append(
        f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" />'
    )

    group_width = plot_width / max(len(task_names), 1)
    inner_pad = group_width * 0.15
    bars_area = group_width - 2 * inner_pad
    bar_width = bars_area / max(len(run_labels), 1)

    for task_idx, task_name in enumerate(task_names):
        group_x = margin_left + task_idx * group_width
        svg.append(
            f'<text class="small" x="{group_x + group_width/2:.1f}" y="{margin_top + plot_height + 42}" text-anchor="middle">{html.escape(task_name)}</text>'
        )
        for run_idx, run_label in enumerate(run_labels):
            value = values_by_run_and_task.get(run_label, {}).get(task_name)
            if value is None:
                continue
            bar_h = (value / max_value) * plot_height
            x = group_x + inner_pad + run_idx * bar_width
            y = margin_top + plot_height - bar_h
            color = colors[run_idx % len(colors)]
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 2, 1):.1f}" height="{bar_h:.1f}" fill="{color}" />'
            )
            svg.append(
                f'<text class="small" x="{x + bar_width/2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:.3f}</text>'
            )

    # legend
    legend_x = margin_left
    legend_y = height - 22
    for idx, run_label in enumerate(run_labels):
        color = colors[idx % len(colors)]
        x = legend_x + idx * 140
        svg.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />')
        svg.append(
            f'<text class="small" x="{x + 18}" y="{legend_y}" text-anchor="start">{html.escape(run_label)}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def build_html(rows, metrics):
    if not rows:
        return "<html><body><h1>No rows found.</h1></body></html>"

    run_labels = unique_in_order(row["run_label"] for row in rows)
    task_names = unique_in_order(row["tasks"] for row in rows)

    sections = [
        "<html><head><meta charset='utf-8'><title>Evaluation Visualization</title></head><body>",
        "<h1>Evaluation Visualization</h1>",
        "<p>Rows shown are filtered to one <code>group_len</code> value in the input script arguments.</p>",
    ]

    sections.append("<h2>Raw Table</h2>")
    sections.append("<table border='1' cellspacing='0' cellpadding='6'>")
    sections.append("<tr><th>run_label</th><th>tasks</th><th>num_samples</th>" + "".join(f"<th>{html.escape(metric)}</th>" for metric in metrics) + "</tr>")
    for row in rows:
        sections.append(
            "<tr>"
            f"<td>{html.escape(row['run_label'])}</td>"
            f"<td>{html.escape(row['tasks'])}</td>"
            f"<td>{html.escape(str(row.get('num_samples', '')))}</td>"
            + "".join(
                f"<td>{html.escape(str(row.get(metric, '')))}</td>" for metric in metrics
            )
            + "</tr>"
        )
    sections.append("</table>")

    for metric in metrics:
        values_by_run_and_task = {}
        for run_label in run_labels:
            values_by_run_and_task[run_label] = {}
            for task_name in task_names:
                matching = next(
                    (
                        row
                        for row in rows
                        if row["run_label"] == run_label and row["tasks"] == task_name
                    ),
                    None,
                )
                values_by_run_and_task[run_label][task_name] = (
                    coerce_number(matching.get(metric)) if matching else None
                )
        sections.append(f"<h2>{html.escape(metric)}</h2>")
        sections.append(
            svg_grouped_bar_chart(metric, task_names, run_labels, values_by_run_and_task)
        )

    sections.append("</body></html>")
    return "\n".join(sections)


def main():
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.run]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_results(run_specs, args.csv_name, args.group_len)
    rows.sort(key=lambda row: (row["tasks"], row["run_label"]))

    write_summary_csv(rows, output_dir / "merged_results.csv")
    write_summary_json(rows, output_dir / "merged_results.json")

    html_report = build_html(rows, args.metrics)
    (output_dir / "report.html").write_text(html_report, encoding="utf-8")

    print(f"Wrote {output_dir / 'merged_results.csv'}")
    print(f"Wrote {output_dir / 'merged_results.json'}")
    print(f"Wrote {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
