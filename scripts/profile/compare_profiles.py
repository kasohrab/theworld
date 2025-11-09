"""
Compare profiling results from multiple runs (e.g., Gemma vs TheWorld).

This script loads profiling_summary.json files from multiple profiling runs
and generates side-by-side comparisons to identify performance bottlenecks.

Usage:
    # Compare two profiling runs
    python scripts/profile/compare_profiles.py \
        checkpoints/profiling/20250108_123456_gemma/ \
        checkpoints/profiling/20250108_234567_theworld/

    # Compare three runs (Gemma, TheWorld projection, TheWorld full)
    python scripts/profile/compare_profiles.py \
        checkpoints/profiling/*_gemma/ \
        checkpoints/profiling/*_theworld_projection/ \
        checkpoints/profiling/*_theworld_full/

    # Save comparison to file
    python scripts/profile/compare_profiles.py \
        checkpoints/profiling/*_gemma/ \
        checkpoints/profiling/*_theworld/ \
        --output comparison_report.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_profiling_summary(profile_dir: str) -> Dict[str, Any]:
    """Load profiling_summary.json from a profiling directory.

    Args:
        profile_dir: Path to profiling directory

    Returns:
        Dictionary with profiling data
    """
    profile_path = Path(profile_dir) / "profiling_summary.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"Profiling summary not found: {profile_path}")

    with open(profile_path, "r") as f:
        return json.load(f)


def format_memory_gb(gb: float) -> str:
    """Format memory in GB with color coding."""
    return f"{gb:.2f} GB"


def format_time_ms(ms: float) -> str:
    """Format time in milliseconds."""
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    elif ms >= 1:
        return f"{ms:.1f}ms"
    else:
        return f"{ms:.3f}ms"


def calculate_speedup(baseline_ms: float, comparison_ms: float) -> str:
    """Calculate speedup/slowdown factor.

    Args:
        baseline_ms: Baseline time in ms
        comparison_ms: Comparison time in ms

    Returns:
        Formatted speedup string (e.g., "2.5x slower", "1.3x faster")
    """
    if baseline_ms == 0 or comparison_ms == 0:
        return "N/A"

    ratio = comparison_ms / baseline_ms
    if ratio > 1.05:  # More than 5% slower
        return f"{ratio:.2f}x slower"
    elif ratio < 0.95:  # More than 5% faster
        return f"{1/ratio:.2f}x faster"
    else:
        return "~same"


def print_section_header(title: str, width: int = 100):
    """Print a formatted section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def compare_model_configs(profiles: List[Dict[str, Any]], labels: List[str]):
    """Compare model configurations across profiling runs."""
    print_section_header("Model Configuration Comparison")

    # Header
    print(f"\n{'Setting':<30}", end="")
    for label in labels:
        print(f"{label:<25}", end="")
    print()
    print("-" * (30 + 25 * len(labels)))

    # Extract configs
    configs = [p.get("model_config", {}) for p in profiles]

    # Compare key settings
    settings = [
        ("Model", "model_name"),
        ("Enable World", "enable_world"),
        ("Cosmos Model", "cosmos_model_name"),
        ("Freeze Gemma Vision", "freeze_gemma_vision"),
        ("Freeze Gemma Language", "freeze_gemma_language"),
        ("Freeze Cosmos VAE", "freeze_cosmos_vae"),
        ("Batch Size", "batch_size"),
        ("Gradient Checkpointing", "use_gradient_checkpointing"),
        ("Mixed Precision", "mixed_precision"),
    ]

    for setting_name, setting_key in settings:
        print(f"{setting_name:<30}", end="")
        for config in configs:
            value = config.get(setting_key, "N/A")
            # Truncate long strings
            value_str = str(value)
            if len(value_str) > 23:
                value_str = value_str[:20] + "..."
            print(f"{value_str:<25}", end="")
        print()


def compare_memory_usage(profiles: List[Dict[str, Any]], labels: List[str]):
    """Compare GPU memory usage across profiling runs."""
    print_section_header("GPU Memory Usage Comparison")

    # Header
    print(f"\n{'Metric':<30}", end="")
    for label in labels:
        print(f"{label:<20}", end="")
    print()
    print("-" * (30 + 20 * len(labels)))

    # Extract memory data
    memory_data = [p.get("gpu_memory", {}) for p in profiles]

    metrics = [
        ("Allocated", "allocated_gb"),
        ("Reserved", "reserved_gb"),
        ("Max Allocated (Peak)", "max_allocated_gb"),
    ]

    for metric_name, metric_key in metrics:
        print(f"{metric_name:<30}", end="")
        values = []
        for mem in memory_data:
            value = mem.get(metric_key, 0)
            values.append(value)
            print(f"{format_memory_gb(value):<20}", end="")
        print()

        # Show relative differences (if more than 1 profile)
        if len(values) > 1:
            baseline = values[0]
            print(f"{'  vs ' + labels[0]:<30}", end="")
            print(f"{'baseline':<20}", end="")
            for i in range(1, len(values)):
                if baseline > 0:
                    diff = ((values[i] - baseline) / baseline) * 100
                    print(f"{diff:+.1f}%{' ' * 13}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()


def compare_cuda_operations(profiles: List[Dict[str, Any]], labels: List[str], top_k: int = 20):
    """Compare top CUDA operations across profiling runs."""
    print_section_header(f"Top {top_k} CUDA Operations by Time")

    # Get all unique operation names across all profiles
    all_ops = set()
    for profile in profiles:
        for op in profile.get("top_cuda_operations", [])[:top_k]:
            all_ops.add(op["name"])

    # Create mapping of op_name -> time for each profile
    op_times = []
    for profile in profiles:
        op_map = {}
        for op in profile.get("top_cuda_operations", []):
            op_map[op["name"]] = op["self_device_time_ms"]
        op_times.append(op_map)

    # Sort operations by baseline (first profile) time
    baseline_ops = sorted(
        op_times[0].items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Header
    print(f"\n{'Operation':<50}", end="")
    for label in labels:
        print(f"{label:<20}", end="")
    if len(labels) > 1:
        print(f"{'Speedup':<15}", end="")
    print()
    print("-" * (50 + 20 * len(labels) + (15 if len(labels) > 1 else 0)))

    # Print operations
    for op_name, baseline_time in baseline_ops:
        # Truncate operation name if too long
        display_name = op_name
        if len(display_name) > 48:
            display_name = display_name[:45] + "..."

        print(f"{display_name:<50}", end="")

        # Print times for each profile
        times = []
        for op_map in op_times:
            time_ms = op_map.get(op_name, 0)
            times.append(time_ms)
            print(f"{format_time_ms(time_ms):<20}", end="")

        # Print speedup (comparison vs baseline)
        if len(times) > 1:
            speedup = calculate_speedup(times[0], times[1])
            print(f"{speedup:<15}", end="")

        print()


def compare_total_times(profiles: List[Dict[str, Any]], labels: List[str]):
    """Compare total CUDA and CPU times across profiling runs."""
    print_section_header("Total Time Breakdown")

    # Calculate total times
    cuda_totals = []
    cpu_totals = []

    for profile in profiles:
        # Sum top 10 CUDA operations as proxy for total
        cuda_total = sum(op["self_device_time_ms"] for op in profile.get("top_cuda_operations", [])[:10])
        cpu_total = sum(op["self_cpu_time_ms"] for op in profile.get("top_cpu_operations", [])[:10])
        cuda_totals.append(cuda_total)
        cpu_totals.append(cpu_total)

    # Print comparison
    print(f"\n{'Metric':<30}", end="")
    for label in labels:
        print(f"{label:<20}", end="")
    if len(labels) > 1:
        print(f"{'Speedup':<15}", end="")
    print()
    print("-" * (30 + 20 * len(labels) + (15 if len(labels) > 1 else 0)))

    # CUDA time
    print(f"{'Top 10 CUDA Time':<30}", end="")
    for time_ms in cuda_totals:
        print(f"{format_time_ms(time_ms):<20}", end="")
    if len(cuda_totals) > 1:
        print(f"{calculate_speedup(cuda_totals[0], cuda_totals[1]):<15}", end="")
    print()

    # CPU time
    print(f"{'Top 10 CPU Time':<30}", end="")
    for time_ms in cpu_totals:
        print(f"{format_time_ms(time_ms):<20}", end="")
    if len(cpu_totals) > 1:
        print(f"{calculate_speedup(cpu_totals[0], cpu_totals[1]):<15}", end="")
    print()


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare profiling results from multiple runs")

    parser.add_argument(
        "profile_dirs",
        nargs="+",
        help="Paths to profiling directories (containing profiling_summary.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save comparison to file (default: print to console)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top operations to compare (default: 20)",
    )

    args = parser.parse_args()

    # Load profiling data
    print("Loading profiling data...")
    profiles = []
    labels = []
    for profile_dir in args.profile_dirs:
        try:
            profile_data = load_profiling_summary(profile_dir)
            profiles.append(profile_data)

            # Generate label from directory name
            label = Path(profile_dir).name
            # Remove timestamp prefix (YYYYMMDD_HHMMSS_)
            if "_" in label:
                parts = label.split("_")
                if len(parts) >= 3:
                    label = "_".join(parts[2:])  # Skip timestamp and job_id
            labels.append(label)

            print(f"  ✓ {profile_dir} ({label})")
        except Exception as e:
            print(f"  ✗ Failed to load {profile_dir}: {e}")
            sys.exit(1)

    print(f"\nComparing {len(profiles)} profiling runs...")

    # Redirect output if requested
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, "w")

    # Generate comparison report
    print("=" * 100)
    print("  PROFILING COMPARISON REPORT")
    print("=" * 100)
    print(f"\nProfiles compared: {len(profiles)}")
    for i, (label, profile_dir) in enumerate(zip(labels, args.profile_dirs)):
        print(f"  [{i+1}] {label}: {profile_dir}")

    # Compare different aspects
    compare_model_configs(profiles, labels)
    compare_memory_usage(profiles, labels)
    compare_total_times(profiles, labels)
    compare_cuda_operations(profiles, labels, top_k=args.top_k)

    # Summary
    print_section_header("Summary")
    if len(profiles) > 1:
        # Compare first two profiles
        cuda_ops_0 = profiles[0].get("top_cuda_operations", [])
        cuda_ops_1 = profiles[1].get("top_cuda_operations", [])

        total_time_0 = sum(op["self_device_time_ms"] for op in cuda_ops_0[:10])
        total_time_1 = sum(op["self_device_time_ms"] for op in cuda_ops_1[:10])

        print(f"\n{labels[1]} vs {labels[0]}:")
        print(f"  Top 10 CUDA time: {calculate_speedup(total_time_0, total_time_1)}")

        mem_0 = profiles[0].get("gpu_memory", {}).get("max_allocated_gb", 0)
        mem_1 = profiles[1].get("gpu_memory", {}).get("max_allocated_gb", 0)
        if mem_0 > 0:
            mem_diff = ((mem_1 - mem_0) / mem_0) * 100
            print(f"  Peak memory: {mem_diff:+.1f}%")

    # Restore stdout
    if args.output:
        sys.stdout = original_stdout
        print(f"\n✓ Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
