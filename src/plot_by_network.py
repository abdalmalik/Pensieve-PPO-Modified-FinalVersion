import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACE_GROUP_PATTERN = re.compile(r"trace_group_(\d+)", re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot average QoE grouped by evaluation trace group.")
    parser.add_argument(
        "--input-dir",
        default="test_results",
        help="Directory containing per-trace evaluation logs.",
    )
    parser.add_argument(
        "--output-path",
        default="final_results/performance_by_trace_group.png",
        help="Path where the summary figure will be saved.",
    )
    return parser.parse_args()


def infer_trace_group(name: str) -> str:
    match = TRACE_GROUP_PATTERN.search(name)
    if match:
        return f"trace_group_{int(match.group(1)):02d}"
    return "trace_group_unknown"


def extract_qoe(log_path: Path) -> float | None:
    lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None
    return float(lines[-1].split()[-1])


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_groups: dict[str, list[float]] = defaultdict(list)
    for log_path in sorted(input_dir.iterdir()):
        if not log_path.is_file():
            continue
        qoe = extract_qoe(log_path)
        if qoe is None:
            continue
        trace_groups[infer_trace_group(log_path.name)].append(qoe)

    if not trace_groups:
        raise SystemExit("No valid evaluation logs found in the input directory.")

    labels = sorted(trace_groups.keys())
    means = [np.mean(trace_groups[label]) for label in labels]
    stds = [np.std(trace_groups[label]) for label in labels]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, means, yerr=stds, capsize=8, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target QoE = 0.75')
    plt.ylabel('Average QoE', fontsize=12)
    plt.xlabel('Trace Group', fontsize=12)
    plt.title('Performance Across Evaluation Trace Groups (Optimized Model)', fontsize=14)
    plt.xticks(rotation=25, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
