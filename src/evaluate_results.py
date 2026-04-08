import argparse
import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACE_GROUP_PATTERN = re.compile(r"trace_group_(\d+)", re.IGNORECASE)


def infer_trace_group(name: str) -> str:
    match = TRACE_GROUP_PATTERN.search(name)
    if match:
        return f"trace_group_{int(match.group(1)):02d}"
    return "trace_group_unknown"


def normalize_trace_name(name: str) -> str:
    trace_name = Path(name).name
    if trace_name.startswith("log_sim_ppo_"):
        trace_name = trace_name[len("log_sim_ppo_"):]
    return trace_name


def parse_log_file(path: Path) -> dict | None:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            time_s, bitrate_kbps, buffer_s, rebuffer_s, chunk_size_bytes, delay_ms, entropy, reward = map(float, parts[:8])
            rows.append(
                {
                    "time_s": time_s,
                    "bitrate_kbps": bitrate_kbps,
                    "buffer_s": buffer_s,
                    "rebuffer_s": rebuffer_s,
                    "chunk_size_bytes": chunk_size_bytes,
                    "delay_ms": delay_ms,
                    "entropy": entropy,
                    "reward": reward,
                }
            )

    if not rows:
        return None

    rewards = [row["reward"] for row in rows]
    bitrates_mbps = [row["bitrate_kbps"] / 1000.0 for row in rows]
    rebuffer_values = [row["rebuffer_s"] for row in rows]
    smoothness_values = [abs(bitrates_mbps[index] - bitrates_mbps[index - 1]) for index in range(1, len(bitrates_mbps))]
    video_duration_s = 4.0 * len(rows)
    total_rebuffer_s = sum(rebuffer_values)

    return {
        "trace": normalize_trace_name(path.name),
        "trace_group": infer_trace_group(path.name),
        "chunks": len(rows),
        "mean_qoe": float(np.mean(rewards[1:] if len(rewards) > 1 else rewards)),
        "sum_qoe": float(np.sum(rewards)),
        "min_qoe": float(np.min(rewards)),
        "max_qoe": float(np.max(rewards)),
        "avg_bitrate_mbps": float(np.mean(bitrates_mbps)),
        "avg_buffer_s": float(np.mean([row["buffer_s"] for row in rows])),
        "total_rebuffer_s": float(total_rebuffer_s),
        "rebuffer_ratio_pct": float((total_rebuffer_s / (video_duration_s + total_rebuffer_s) * 100.0) if (video_duration_s + total_rebuffer_s) else 0.0),
        "avg_smoothness_mbps": float(np.mean(smoothness_values) if smoothness_values else 0.0),
    }


def build_summary(trace_metrics: list[dict], model_path: str) -> dict:
    qoes = np.array([item["mean_qoe"] for item in trace_metrics], dtype=float)
    bitrates = np.array([item["avg_bitrate_mbps"] for item in trace_metrics], dtype=float)
    rebuffer_totals = np.array([item["total_rebuffer_s"] for item in trace_metrics], dtype=float)
    rebuffer_ratios = np.array([item["rebuffer_ratio_pct"] for item in trace_metrics], dtype=float)
    smoothness = np.array([item["avg_smoothness_mbps"] for item in trace_metrics], dtype=float)
    buffers = np.array([item["avg_buffer_s"] for item in trace_metrics], dtype=float)

    return {
        "model_path": model_path,
        "trace_count": len(trace_metrics),
        "qoe": {
            "mean": float(np.mean(qoes)),
            "median": float(np.median(qoes)),
            "std": float(np.std(qoes)),
            "min": float(np.min(qoes)),
            "p5": float(np.percentile(qoes, 5)),
            "p95": float(np.percentile(qoes, 95)),
            "max": float(np.max(qoes)),
        },
        "bitrate": {
            "avg_mbps": float(np.mean(bitrates)),
            "min_mbps": float(np.min(bitrates)),
            "max_mbps": float(np.max(bitrates)),
        },
        "rebuffer": {
            "avg_total_seconds": float(np.mean(rebuffer_totals)),
            "median_total_seconds": float(np.median(rebuffer_totals)),
            "avg_ratio_pct": float(np.mean(rebuffer_ratios)),
        },
        "smoothness": {
            "avg_mbps": float(np.mean(smoothness)),
        },
        "buffer": {
            "avg_seconds": float(np.mean(buffers)),
        },
    }


def build_trace_group_summary(trace_metrics: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for metric in trace_metrics:
        grouped[metric["trace_group"]].append(metric)

    trace_group_rows = []
    for trace_group in sorted(grouped.keys()):
        entries = grouped[trace_group]
        trace_group_rows.append(
            {
                "trace_group": trace_group,
                "trace_count": len(entries),
                "mean_qoe": float(np.mean([entry["mean_qoe"] for entry in entries])),
                "avg_bitrate_mbps": float(np.mean([entry["avg_bitrate_mbps"] for entry in entries])),
                "avg_total_rebuffer_s": float(np.mean([entry["total_rebuffer_s"] for entry in entries])),
                "avg_rebuffer_ratio_pct": float(np.mean([entry["rebuffer_ratio_pct"] for entry in entries])),
                "avg_smoothness_mbps": float(np.mean([entry["avg_smoothness_mbps"] for entry in entries])),
            }
        )
    return trace_group_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_text(path: Path, summary: dict, trace_group_rows: list[dict]) -> None:
    lines = [
        "Pensieve PPO Evaluation Summary",
        "=" * 40,
        f"Model: {summary['model_path']}",
        f"Trace count: {summary['trace_count']}",
        "",
        "QoE",
        f"  Mean   : {summary['qoe']['mean']:.6f}",
        f"  Median : {summary['qoe']['median']:.6f}",
        f"  Std    : {summary['qoe']['std']:.6f}",
        f"  Min    : {summary['qoe']['min']:.6f}",
        f"  P5     : {summary['qoe']['p5']:.6f}",
        f"  P95    : {summary['qoe']['p95']:.6f}",
        f"  Max    : {summary['qoe']['max']:.6f}",
        "",
        "Bitrate",
        f"  Avg Mbps: {summary['bitrate']['avg_mbps']:.6f}",
        f"  Min Mbps: {summary['bitrate']['min_mbps']:.6f}",
        f"  Max Mbps: {summary['bitrate']['max_mbps']:.6f}",
        "",
        "Rebuffer",
        f"  Avg total seconds : {summary['rebuffer']['avg_total_seconds']:.6f}",
        f"  Median total secs : {summary['rebuffer']['median_total_seconds']:.6f}",
        f"  Avg ratio percent : {summary['rebuffer']['avg_ratio_pct']:.6f}",
        "",
        "Smoothness",
        f"  Avg Mbps: {summary['smoothness']['avg_mbps']:.6f}",
        "",
        "Average buffer",
        f"  Avg seconds: {summary['buffer']['avg_seconds']:.6f}",
        "",
        "By trace group",
    ]

    for row in trace_group_rows:
        lines.append(
            f"  {row['trace_group']}: traces={row['trace_count']}, "
            f"mean_qoe={row['mean_qoe']:.6f}, avg_bitrate={row['avg_bitrate_mbps']:.6f}, "
            f"avg_rebuffer={row['avg_total_rebuffer_s']:.6f}, avg_smoothness={row['avg_smoothness_mbps']:.6f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_qoe_distribution(trace_metrics: list[dict], output_path: Path) -> None:
    qoe_values = [item["mean_qoe"] for item in trace_metrics]
    plt.figure(figsize=(10, 6))
    plt.hist(qoe_values, bins=24, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(np.mean(qoe_values), color="crimson", linestyle="--", linewidth=2, label=f"Mean QoE = {np.mean(qoe_values):.3f}")
    plt.axvline(0.75, color="darkgreen", linestyle=":", linewidth=2, label="Target QoE = 0.75")
    plt.xlabel("Mean QoE per trace")
    plt.ylabel("Frequency")
    plt.title("QoE Distribution for Evaluated Test Traces")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_trace_group_qoe(trace_group_rows: list[dict], output_path: Path) -> None:
    labels = [item["trace_group"] for item in trace_group_rows]
    values = [item["mean_qoe"] for item in trace_group_rows]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="teal", alpha=0.8, edgecolor="black")
    plt.axhline(0.75, color="crimson", linestyle="--", linewidth=2, label="Target QoE = 0.75")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xlabel("Trace group")
    plt.ylabel("Mean QoE")
    plt.title("Average QoE by Trace Group")
    plt.grid(alpha=0.25, axis="y")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_trace_group_rebuffer(trace_group_rows: list[dict], output_path: Path) -> None:
    labels = [item["trace_group"] for item in trace_group_rows]
    values = [item["avg_total_rebuffer_s"] for item in trace_group_rows]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="darkorange", alpha=0.8, edgecolor="black")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xlabel("Trace group")
    plt.ylabel("Average total rebuffer seconds")
    plt.title("Average Rebuffer by Trace Group")
    plt.grid(alpha=0.25, axis="y")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def copy_raw_logs(src_dir: Path, output_dir: Path) -> None:
    raw_dir = output_dir / "raw_test_logs"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    shutil.copytree(src_dir, raw_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Pensieve test_results logs.")
    parser.add_argument("--input-dir", required=True, help="Directory containing test log files.")
    parser.add_argument("--output-dir", required=True, help="Directory where reports will be saved.")
    parser.add_argument("--model-path", required=True, help="Model checkpoint used to generate the test logs.")
    parser.add_argument("--copy-logs", action="store_true", help="Copy raw test logs into the output directory.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_metrics = []
    for log_file in sorted(input_dir.iterdir()):
        if not log_file.is_file():
            continue
        parsed = parse_log_file(log_file)
        if parsed is not None:
            trace_metrics.append(parsed)

    if not trace_metrics:
        raise SystemExit("No valid test logs found in the input directory.")

    summary = build_summary(trace_metrics, args.model_path)
    trace_group_rows = build_trace_group_summary(trace_metrics)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_text(output_dir / "summary.txt", summary, trace_group_rows)
    write_csv(output_dir / "per_trace_metrics.csv", trace_metrics)
    write_csv(output_dir / "by_trace_group.csv", trace_group_rows)
    plot_qoe_distribution(trace_metrics, output_dir / "qoe_distribution.png")
    plot_trace_group_qoe(trace_group_rows, output_dir / "qoe_by_trace_group.png")
    plot_trace_group_rebuffer(trace_group_rows, output_dir / "rebuffer_by_trace_group.png")

    if args.copy_logs:
        copy_raw_logs(input_dir, output_dir)

    print(f"Saved evaluation outputs to: {output_dir}")


if __name__ == "__main__":
    main()
