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
        description="Visualize recent-position queue experiment results."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the form LABEL=DIR_OR_CSV.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv-name", default="queue_position_results.csv")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run value: {spec!r}. Expected LABEL=DIR_OR_CSV.")
    label, raw_path = spec.split("=", 1)
    return label.strip(), Path(raw_path.strip()).expanduser().resolve()


def resolve_csv_path(path: Path, csv_name: str) -> Path:
    return path if path.is_file() else path / csv_name


def coerce_float(value):
    if value in {"", None, "None", "N/A"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_rows(run_specs, csv_name: str):
    rows = []
    for label, raw_path in run_specs:
        csv_path = resolve_csv_path(raw_path, csv_name)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find queue position CSV: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["run_label"] = label
                row["run_path"] = str(raw_path)
                row["recent_position"] = int(row["recent_position"])
                rows.append(row)
    return rows


def write_csv(rows, path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def svg_grouped_bar_chart(title: str, x_values, series_map):
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
        for series in series_map.values()
        for value in series.values()
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

    group_width = plot_width / max(len(x_values), 1)
    inner_pad = group_width * 0.15
    bars_area = group_width - 2 * inner_pad
    bar_width = bars_area / max(len(series_map), 1)

    for x_idx, x_value in enumerate(x_values):
        group_x = margin_left + x_idx * group_width
        svg.append(
            f'<text class="small" x="{group_x + group_width/2:.1f}" y="{margin_top + plot_height + 42}" text-anchor="middle">{html.escape(str(x_value))}</text>'
        )
        for series_idx, (series_name, series) in enumerate(series_map.items()):
            value = series.get(x_value)
            if value is None:
                continue
            bar_h = (value / max_value) * plot_height
            x = group_x + inner_pad + series_idx * bar_width
            y = margin_top + plot_height - bar_h
            color = colors[series_idx % len(colors)]
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_width - 2, 1):.1f}" height="{bar_h:.1f}" fill="{color}" />'
            )
            svg.append(
                f'<text class="small" x="{x + bar_width/2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{value:.3f}</text>'
            )

    legend_x = margin_left
    legend_y = 32
    for idx, series_name in enumerate(series_map.keys()):
        color = colors[idx % len(colors)]
        x = legend_x + idx * 150
        svg.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />')
        svg.append(
            f'<text class="small" x="{x + 18}" y="{legend_y}" text-anchor="start">{html.escape(series_name)}</text>'
        )

    svg.append(
        f'<text class="small" x="{margin_left + plot_width/2:.1f}" y="{height - 10}" text-anchor="middle">recent_position</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def render_table(rows):
    if not rows:
        return "<p>No rows loaded.</p>"
    fieldnames = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(name)}</th>" for name in fieldnames)
    body_rows = []
    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(name, '')))}</td>" for name in fieldnames
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )


def build_report(rows, metrics):
    datasets = unique_in_order(row["dataset"] for row in rows)
    run_labels = unique_in_order(row["run_label"] for row in rows)
    recent_positions = sorted(unique_in_order(row["recent_position"] for row in rows))

    charts = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        charts.append(f"<h2>{html.escape(dataset)}</h2>")
        for metric in metrics:
            series_map = {}
            for run_label in run_labels:
                run_rows = [
                    row for row in dataset_rows if row["run_label"] == run_label
                ]
                series_map[run_label] = {
                    row["recent_position"]: coerce_float(row.get(metric))
                    for row in run_rows
                }
            charts.append(
                svg_grouped_bar_chart(
                    f"{dataset} - {metric}",
                    recent_positions,
                    series_map,
                )
            )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>LoRA Queue Position Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #fafafa;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .chart {{
      background: white;
      padding: 12px;
      margin: 16px 0;
      border: 1px solid #ddd;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 6px 8px;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #f0f0f0;
    }}
  </style>
</head>
<body>
  <h1>LoRA Queue Position Evaluation Report</h1>
  <p>Runs: {html.escape(", ".join(run_labels))}</p>
  <p>Datasets: {html.escape(", ".join(datasets))}</p>
  <h2>Merged Results</h2>
  {render_table(rows)}
  {"".join(f'<div class="chart">{chart}</div>' for chart in charts)}
</body>
</html>
"""
    return html_doc


def main():
    args = parse_args()
    run_specs = [parse_run_spec(spec) for spec in args.run]
    rows = load_rows(run_specs, args.csv_name)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, output_dir / "merged_queue_position_results.csv")
    write_json(rows, output_dir / "merged_queue_position_results.json")
    (output_dir / "report.html").write_text(
        build_report(rows, args.metrics),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
