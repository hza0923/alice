from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "outputs" / "runtime" / "logs"
OUT_DIR = ROOT / "outputs" / "runtime"

METRICS = ["throughput_token_s", "avg_ttft_s", "avg_e2e_s"]


def load_summary(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "summary" in payload:
        return payload["summary"]
    if "pipeline" in payload:
        pipeline = payload["pipeline"]
        return {
            "throughput_token_s": float(pipeline.get("throughput_tokens_per_s", 0.0)),
            "avg_ttft_s": 0.0,
            "avg_e2e_s": 0.0,
        }
    raise KeyError(f"No summary found in {metrics_path}")


def collect_len_data() -> tuple[list[int], dict[str, list[float]], dict[str, list[float]]]:
    fine_dirs = sorted(LOGS_DIR.glob("fine-grained-bs8-len*"))
    len_values: list[int] = []
    fine_series = {m: [] for m in METRICS}
    coarse_series = {m: [] for m in METRICS}

    for fine_dir in fine_dirs:
        match = re.search(r"len(\d+)$", fine_dir.name)
        if not match:
            continue
        seq_len = int(match.group(1))
        coarse_dir = LOGS_DIR / fine_dir.name.replace("fine-grained", "coarse-grained")
        fine_metrics = fine_dir / "3060_0.metrics.json"
        coarse_metrics = coarse_dir / "3060_0.metrics.json"
        if not fine_metrics.exists() or not coarse_metrics.exists():
            continue

        fine_summary = load_summary(fine_metrics)
        coarse_summary = load_summary(coarse_metrics)
        len_values.append(seq_len)
        for metric in METRICS:
            fine_series[metric].append(float(fine_summary[metric]))
            coarse_series[metric].append(float(coarse_summary[metric]))

    order = sorted(range(len(len_values)), key=len_values.__getitem__)
    len_values = [len_values[i] for i in order]
    for metric in METRICS:
        fine_series[metric] = [fine_series[metric][i] for i in order]
        coarse_series[metric] = [coarse_series[metric][i] for i in order]
    return len_values, fine_series, coarse_series


def plot_len_comparison() -> Path:
    lengths, fine_series, coarse_series = collect_len_data()
    x = range(len(lengths))
    width = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    titles = {
        "throughput_token_s": "Throughput (token/s)",
        "avg_ttft_s": "Avg TTFT (s)",
        "avg_e2e_s": "Avg E2E (s)",
    }
    for ax, metric in zip(axes, METRICS):
        ax.bar([i - width / 2 for i in x], fine_series[metric], width=width, label="fine-grained")
        ax.bar([i + width / 2 for i in x], coarse_series[metric], width=width, label="coarse-grained")
        ax.set_title(titles[metric])
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(v) for v in lengths], rotation=0)
        ax.set_xlabel("len")
        ax.grid(axis="y", alpha=0.25)
        if metric == "throughput_token_s":
            ax.set_ylabel("token/s")
        else:
            ax.set_ylabel("seconds")

    axes[0].legend(loc="best")
    fig.suptitle("BS8 len sweep: fine vs coarse", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "fine_vs_coarse_len_metrics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_harmonic_comparison() -> Path:
    fine_summary = load_summary(LOGS_DIR / "fine-grained-bs8-harmonic_reasoning_v1" / "3060_0.metrics.json")
    coarse_summary = load_summary(LOGS_DIR / "coarse-grained-bs8-harmonic_reasoning_v1" / "3060_0.metrics.json")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    throughput_values = [fine_summary["throughput_token_s"], coarse_summary["throughput_token_s"]]
    axes[0].bar(["fine-grained", "coarse-grained"], throughput_values, color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Throughput (token/s)")
    axes[0].set_ylabel("token/s")
    axes[0].grid(axis="y", alpha=0.25)
    for idx, value in enumerate(throughput_values):
        axes[0].text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    quantile_labels = ["avg", "p50", "p90", "p95", "p99"]
    ttft_keys = ["avg_ttft_s", "p50_ttft_s", "p90_ttft_s", "p95_ttft_s", "p99_ttft_s"]
    e2e_keys = ["avg_e2e_s", "p50_e2e_s", "p90_e2e_s", "p95_e2e_s", "p99_e2e_s"]

    fine_ttft = [float(fine_summary[k]) for k in ttft_keys]
    coarse_ttft = [float(coarse_summary[k]) for k in ttft_keys]
    fine_e2e = [float(fine_summary[k]) for k in e2e_keys]
    coarse_e2e = [float(coarse_summary[k]) for k in e2e_keys]

    axes[1].plot(quantile_labels, fine_ttft, marker="o", linewidth=2, label="fine-grained")
    axes[1].plot(quantile_labels, coarse_ttft, marker="s", linewidth=2, label="coarse-grained")
    axes[1].set_title("TTFT Quantiles (s)")
    axes[1].set_ylabel("seconds")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(quantile_labels, fine_e2e, marker="o", linewidth=2, label="fine-grained")
    axes[2].plot(quantile_labels, coarse_e2e, marker="s", linewidth=2, label="coarse-grained")
    axes[2].set_title("E2E Quantiles (s)")
    axes[2].set_ylabel("seconds")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Harmonic reasoning v1: real-request performance", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "fine_vs_coarse_harmonic_reasoning_metrics.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path1 = plot_len_comparison()
    path2 = plot_harmonic_comparison()
    print(f"Saved: {path1}")
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
