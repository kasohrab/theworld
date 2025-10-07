"""
Compare evaluation results across different models.

This script compares TheWorld against baselines (Gemma3, random projection, etc.)
to measure the contribution of the world model.

Example usage:
    # Compare TheWorld vs Gemma baseline
    python scripts/compare_results.py \
        --theworld results/theworld_blink.json \
        --baseline results/gemma_baseline_blink.json \
        --output results/comparison.md

    # Compare multiple configurations
    python scripts/compare_results.py \
        --results results/theworld_blink.json results/baseline_blink.json results/random_proj_blink.json \
        --labels "TheWorld" "Gemma3" "Random-Proj" \
        --output results/comparison.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys


def load_result(path: str) -> Dict[str, Any]:
    """Load evaluation result from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from result."""
    metrics = {}

    # Extract from each task and configuration
    for task, task_results in result.get("results", {}).items():
        for config, config_metrics in task_results.items():
            key = f"{task}_{config}"
            metrics[key] = {
                "accuracy": config_metrics.get("accuracy", 0.0),
                "f1_macro": config_metrics.get("f1_macro", 0.0),
                "f1_weighted": config_metrics.get("f1_weighted", 0.0),
                "correct": config_metrics.get("correct", 0),
                "total": config_metrics.get("total", 0),
            }

    # Add summary metrics
    if "summary" in result:
        metrics["summary"] = result["summary"]

    return metrics


def compute_delta(baseline: float, model: float) -> tuple:
    """Compute absolute and percentage delta."""
    abs_delta = model - baseline
    pct_delta = (abs_delta / baseline * 100) if baseline > 0 else 0.0
    return abs_delta, pct_delta


def generate_markdown_report(
    results: List[Dict[str, Any]],
    labels: List[str],
    output_path: Optional[str] = None,
) -> str:
    """Generate markdown comparison report."""
    lines = []

    lines.append("# Model Comparison Report")
    lines.append("")
    lines.append("## Models Evaluated")
    lines.append("")
    for i, (label, result) in enumerate(zip(labels, results)):
        model_name = result.get("model", "Unknown")
        lines.append(f"{i+1}. **{label}**: `{model_name}`")
    lines.append("")

    # Extract all metrics
    all_metrics = [extract_metrics(r) for r in results]

    # Find common tasks/configs
    common_keys = set(all_metrics[0].keys()) - {"summary"}
    for m in all_metrics[1:]:
        common_keys &= set(m.keys()) - {"summary"}

    if not common_keys:
        lines.append("⚠ No common evaluation tasks found across models.")
        lines.append("")
        report = "\n".join(lines)
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
        return report

    # Sort keys for consistent ordering
    common_keys = sorted(common_keys)

    # Create comparison table
    lines.append("## Results Comparison")
    lines.append("")

    for key in common_keys:
        # Parse task and config
        parts = key.rsplit("_", 2)
        if len(parts) >= 3:
            task = "_".join(parts[:-2])
            config = f"{parts[-2]}_{parts[-1]}"
        else:
            task = key
            config = "default"

        lines.append(f"### {task} ({config})")
        lines.append("")

        # Table header
        lines.append("| Metric | " + " | ".join(labels) + " | Δ vs " + labels[0] + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(labels)) + "|--------|")

        # Accuracy row
        acc_values = [m[key]["accuracy"] for m in all_metrics]
        acc_row = "| Accuracy | "
        acc_row += " | ".join([f"{v:.2f}%" for v in acc_values])

        if len(acc_values) > 1:
            _, pct = compute_delta(acc_values[0], acc_values[1])
            delta_str = f"{pct:+.2f}%"
            if pct > 0:
                delta_str = f"✅ {delta_str}"
            elif pct < -5:
                delta_str = f"❌ {delta_str}"
            acc_row += f" | {delta_str} |"
        else:
            acc_row += " | - |"

        lines.append(acc_row)

        # F1 Macro row
        f1_values = [m[key]["f1_macro"] for m in all_metrics]
        f1_row = "| F1 Macro | "
        f1_row += " | ".join([f"{v:.2f}%" for v in f1_values])

        if len(f1_values) > 1:
            _, pct = compute_delta(f1_values[0], f1_values[1])
            f1_row += f" | {pct:+.2f}% |"
        else:
            f1_row += " | - |"

        lines.append(f1_row)

        # Sample counts
        total = all_metrics[0][key]["total"]
        correct_values = [m[key]["correct"] for m in all_metrics]
        count_row = "| Correct | "
        count_row += " | ".join([f"{c}/{total}" for c in correct_values])
        count_row += " | - |"
        lines.append(count_row)

        lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")

    # Overall accuracy comparison
    if "summary" in all_metrics[0]:
        lines.append("### Mean Accuracy Across All Tasks")
        lines.append("")
        mean_accs = [m.get("summary", {}).get("mean_accuracy", 0.0) for m in all_metrics]
        lines.append("| Model | Mean Accuracy | Δ vs Baseline |")
        lines.append("|-------|---------------|---------------|")
        for i, (label, acc) in enumerate(zip(labels, mean_accs)):
            if i == 0:
                lines.append(f"| {label} | {acc:.2f}% | - |")
            else:
                _, pct = compute_delta(mean_accs[0], acc)
                delta_str = f"{pct:+.2f}%"
                if pct > 5:
                    delta_str = f"✅ {delta_str}"
                elif pct < -5:
                    delta_str = f"❌ {delta_str}"
                lines.append(f"| {label} | {acc:.2f}% | {delta_str} |")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    if len(results) >= 2 and "summary" in all_metrics[0] and "summary" in all_metrics[1]:
        baseline_acc = all_metrics[0]["summary"]["mean_accuracy"]
        model_acc = all_metrics[1]["summary"]["mean_accuracy"]
        _, improvement = compute_delta(baseline_acc, model_acc)

        if improvement > 10:
            lines.append(f"✅ **{labels[1]} significantly outperforms {labels[0]}** ({improvement:+.1f}%)")
        elif improvement > 5:
            lines.append(f"✅ **{labels[1]} moderately outperforms {labels[0]}** ({improvement:+.1f}%)")
        elif improvement > 0:
            lines.append(f"⚠ **{labels[1]} slightly outperforms {labels[0]}** ({improvement:+.1f}%)")
        elif improvement > -5:
            lines.append(f"⚠ **{labels[1]} performs similarly to {labels[0]}** ({improvement:+.1f}%)")
        else:
            lines.append(f"❌ **{labels[1]} underperforms {labels[0]}** ({improvement:+.1f}%)")

        lines.append("")
        lines.append("**Interpretation:**")
        if improvement > 5:
            lines.append("- World model provides measurable benefits")
            lines.append("- Continue with this architecture")
        elif improvement > 0:
            lines.append("- Modest improvement - consider:")
            lines.append("  - Longer training")
            lines.append("  - Unfreezing more components")
            lines.append("  - Larger projection layer")
        else:
            lines.append("- World model not helping - investigate:")
            lines.append("  - Check projection layer gradients")
            lines.append("  - Verify world embeddings are being used")
            lines.append("  - Consider different datasets")

    lines.append("")

    # Create report
    report = "\n".join(lines)

    # Save to file if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"✓ Comparison report saved to: {output_path}")

    return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare evaluation results")

    # Option 1: Named baseline vs theworld
    parser.add_argument(
        "--theworld",
        type=str,
        help="Path to TheWorld results JSON",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline results JSON",
    )

    # Option 2: Multiple results with custom labels
    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        help="Paths to result JSON files",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each result file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for markdown report",
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Print report to stdout",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Determine which input mode
    if args.results:
        # Multiple results mode
        result_paths = args.results
        if args.labels:
            if len(args.labels) != len(args.results):
                print("❌ Error: Number of labels must match number of results")
                sys.exit(1)
            labels = args.labels
        else:
            # Generate labels from filenames
            labels = [Path(p).stem for p in result_paths]

    elif args.theworld and args.baseline:
        # Baseline comparison mode
        result_paths = [args.baseline, args.theworld]
        labels = ["Baseline", "TheWorld"]

    else:
        print("❌ Error: Must provide either --theworld + --baseline OR --results")
        print("Usage examples:")
        print("  python compare_results.py --theworld results/tw.json --baseline results/gemma.json")
        print("  python compare_results.py --results results/a.json results/b.json --labels 'Model A' 'Model B'")
        sys.exit(1)

    # Load all results
    print(f"Loading {len(result_paths)} result files...")
    results = []
    for path in result_paths:
        try:
            result = load_result(path)
            results.append(result)
            print(f"  ✓ {path}")
        except Exception as e:
            print(f"  ❌ Failed to load {path}: {e}")
            sys.exit(1)

    # Generate report
    print("\nGenerating comparison report...")
    report = generate_markdown_report(results, labels, args.output)

    # Print if requested
    if args.print or not args.output:
        print("\n" + "=" * 80)
        print(report)
        print("=" * 80)


if __name__ == "__main__":
    main()
