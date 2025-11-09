"""
Profiling script for TheWorld training - runs ONE iteration with comprehensive profiling.

This script profiles:
- CPU usage (operations, kernel launches)
- GPU usage (CUDA kernels, memory transfers)
- Memory allocations (peak, allocations, deallocations)
- Stack traces (exact bottleneck locations)
- Tensor shapes

Usage:
    # Basic profiling (2 samples, projection only)
    python scripts/profile_training.py --config configs/profile.json

    # Profile with gradient checkpointing
    python scripts/profile_training.py --config configs/profile.json --gradient-checkpointing

    # Profile forward pass only (no backward)
    python scripts/profile_training.py --config configs/profile.json --forward-only

    # Adjust number of profiling steps
    python scripts/profile_training.py --config configs/profile.json --profile-steps 5

Output:
    - TensorBoard traces: checkpoints/profiling/traces/
    - Chrome trace: checkpoints/profiling/traces/trace.json
    - Structured data: checkpoints/profiling/*.json, *.csv
    - Console summary: Top operations by CPU/CUDA time and memory
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld, TrainingConfig, create_theworld_collator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile TheWorld training iteration")

    # Default output directory (will be updated with model type after config is loaded)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_dir = f"checkpoints/profiling/{timestamp}"

    parser.add_argument(
        "--config",
        type=str,
        default="configs/profile/profile.json",
        help="Path to training config JSON file",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing (reduces memory, increases compute)",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Profile forward pass only (no backward)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples to profile (default: 2)",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=3,
        help="Number of steps to actively profile (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Output directory for profiling results (default: timestamped directory in checkpoints/profiling/)",
    )

    return parser.parse_args()


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return TrainingConfig.from_dict(config_dict)


def save_profiling_data(prof, output_dir: str, config: TrainingConfig, enable_world: bool, top_k: int = 100):
    """Save profiling data to structured files (JSON and CSV).

    Args:
        prof: PyTorch profiler object
        output_dir: Directory to save files
        config: Training configuration
        enable_world: Whether world model is enabled
        top_k: Number of top operations to save (default: 100)
    """
    # Get profiling data
    key_averages = prof.key_averages()

    # 1. Save top operations by CUDA time (CSV)
    cuda_ops_path = f"{output_dir}/top_cuda_operations.csv"
    with open(cuda_ops_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "self_cuda_time_total_ms",
                "cuda_time_total_ms",
                "num_calls",
                "self_cuda_memory_mb",
                "cuda_memory_mb",
            ]
        )

        # Sort by device (CUDA) time
        sorted_ops = sorted(key_averages, key=lambda x: x.device_time_total, reverse=True)[:top_k]
        for op in sorted_ops:
            writer.writerow(
                [
                    op.key,
                    op.self_device_time_total / 1000,  # Convert to ms
                    op.device_time_total / 1000,
                    op.count,
                    op.self_device_memory_usage / (1024**2),  # Convert to MB
                    op.device_memory_usage / (1024**2),
                ]
            )

    # 2. Save top operations by CPU time (CSV)
    cpu_ops_path = f"{output_dir}/top_cpu_operations.csv"
    with open(cpu_ops_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "self_cpu_time_total_ms",
                "cpu_time_total_ms",
                "num_calls",
                "self_cpu_memory_mb",
                "cpu_memory_mb",
            ]
        )

        # Sort by CPU time
        sorted_ops = sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:top_k]
        for op in sorted_ops:
            writer.writerow(
                [
                    op.key,
                    op.self_cpu_time_total / 1000,  # Convert to ms
                    op.cpu_time_total / 1000,
                    op.count,
                    op.self_cpu_memory_usage / (1024**2),  # Convert to MB
                    op.cpu_memory_usage / (1024**2),
                ]
            )

    # 3. Save top operations by memory (CSV)
    mem_ops_path = f"{output_dir}/top_memory_operations.csv"
    with open(mem_ops_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "self_cuda_memory_mb",
                "cuda_memory_mb",
                "self_cpu_memory_mb",
                "cpu_memory_mb",
                "num_calls",
            ]
        )

        # Sort by device (GPU) memory
        sorted_ops = sorted(key_averages, key=lambda x: abs(x.self_device_memory_usage), reverse=True)[:top_k]
        for op in sorted_ops:
            writer.writerow(
                [
                    op.key,
                    op.self_device_memory_usage / (1024**2),  # Convert to MB
                    op.device_memory_usage / (1024**2),
                    op.self_cpu_memory_usage / (1024**2),
                    op.cpu_memory_usage / (1024**2),
                    op.count,
                ]
            )

    # 4. Save comprehensive JSON with all data
    json_path = f"{output_dir}/profiling_summary.json"
    summary_data = {
        "model_config": {
            "model_name": config.model_name,
            "enable_world": enable_world,
            "cosmos_model_name": config.cosmos_model_name if enable_world else None,
            "freeze_gemma_vision": config.freeze_gemma_vision,
            "freeze_gemma_language": config.freeze_gemma_language,
            "freeze_cosmos_vae": config.freeze_cosmos_vae,
            "batch_size": config.batch_size,
            "use_gradient_checkpointing": config.use_gradient_checkpointing,
            "mixed_precision": config.mixed_precision,
        },
        "gpu_memory": {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0,
        },
        "top_cuda_operations": [
            {
                "name": op.key,
                "self_device_time_ms": op.self_device_time_total / 1000,
                "device_time_total_ms": op.device_time_total / 1000,
                "device_time_avg_ms": (op.device_time_total / op.count / 1000) if op.count > 0 else 0,
                "num_calls": op.count,
                "self_device_memory_mb": op.self_device_memory_usage / (1024**2),
                "device_memory_mb": op.device_memory_usage / (1024**2),
                "flops": op.flops if hasattr(op, "flops") else 0,
            }
            for op in sorted(key_averages, key=lambda x: x.device_time_total, reverse=True)[:top_k]
        ],
        "top_cpu_operations": [
            {
                "name": op.key,
                "self_cpu_time_ms": op.self_cpu_time_total / 1000,
                "cpu_time_total_ms": op.cpu_time_total / 1000,
                "cpu_time_avg_ms": (op.cpu_time_total / op.count / 1000) if op.count > 0 else 0,
                "num_calls": op.count,
                "self_cpu_memory_mb": op.self_cpu_memory_usage / (1024**2),
                "cpu_memory_mb": op.cpu_memory_usage / (1024**2),
            }
            for op in sorted(key_averages, key=lambda x: x.cpu_time_total, reverse=True)[:top_k]
        ],
        "top_memory_operations": [
            {
                "name": op.key,
                "self_device_memory_mb": op.self_device_memory_usage / (1024**2),
                "device_memory_mb": op.device_memory_usage / (1024**2),
                "self_cpu_memory_mb": op.self_cpu_memory_usage / (1024**2),
                "cpu_memory_mb": op.cpu_memory_usage / (1024**2),
                "num_calls": op.count,
            }
            for op in sorted(key_averages, key=lambda x: abs(x.self_device_memory_usage), reverse=True)[:top_k]
        ],
    }

    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"✓ Saved profiling data:")
    print(f"  - CUDA operations: {cuda_ops_path}")
    print(f"  - CPU operations: {cpu_ops_path}")
    print(f"  - Memory operations: {mem_ops_path}")
    print(f"  - JSON summary: {json_path}")
    print()


def load_dataset(config: TrainingConfig, num_samples: int):
    """Load dataset for profiling.

    Args:
        config: TrainingConfig with dataset settings
        num_samples: Number of samples to load

    Returns:
        Dataset instance
    """
    from theworld.datasets import load_spatial_rgpt

    # Authenticate with HuggingFace if token provided
    if config.hf_token:
        try:
            import huggingface_hub

            huggingface_hub.login(token=config.hf_token, add_to_git_credential=False)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            print(f"⚠ HuggingFace authentication failed: {e}")

    if config.dataset_name == "spatial_rgpt":
        from theworld.datasets import SpatialRGPTDataset

        print(f"Loading SpatialRGPT dataset ({num_samples} samples)...")

        # Get image folder from config
        image_folder = getattr(config, "image_folder", None)
        if image_folder is None:
            raise ValueError(
                "image_folder is required for spatial_rgpt dataset. "
                "Set it in your config file to point to OpenImagesV7 directory."
            )

        # Check if train_dataset_path is a local JSON file
        train_path = config.train_dataset_path or "a8cheng/OpenSpatialDataset"

        if os.path.exists(train_path):
            # Local JSON file
            print(f"  Loading from local JSON: {train_path}")
            dataset = SpatialRGPTDataset(
                data_source=train_path,
                image_folder=image_folder,
                draw_bboxes=False,
                num_samples=num_samples,
            )
        else:
            # HuggingFace dataset
            print(f"  Loading from HuggingFace: {train_path}")
            dataset = load_spatial_rgpt(
                split="train",
                image_folder=image_folder,
                num_samples=num_samples,
                draw_bboxes=False,
                hf_token=config.hf_token,
            )

    else:
        raise ValueError(f"Unsupported dataset for profiling: {config.dataset_name}. " f"Use 'spatial_rgpt' in config.")

    return dataset


def main():
    """Main profiling function."""
    args = parse_args()

    print("=" * 80)
    print("TheWorld Training Profiler - Single Iteration Analysis")
    print("=" * 80)
    print()

    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.gradient_checkpointing:
        config.use_gradient_checkpointing = True

    # Get HF token from environment if not in config
    if not config.hf_token:
        config.hf_token = os.environ.get("HF_TOKEN")

    # Get enable_world from config (default to True for backward compatibility)
    enable_world = getattr(config, "enable_world", True)

    # Update output directory with model type if using default path
    if args.output_dir == f"checkpoints/profiling/{datetime.now().strftime('%Y%m%d_%H%M%S')}":
        job_id = os.environ.get("SLURM_JOB_ID", "")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = "gemma" if not enable_world else "theworld"
        if job_id:
            args.output_dir = f"checkpoints/profiling/{timestamp}_{job_id}_{model_type}"
        else:
            args.output_dir = f"checkpoints/profiling/{timestamp}_{model_type}"

    print("Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Enable world model: {enable_world}")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Profile steps: {args.profile_steps}")
    print(f"  Forward only: {args.forward_only}")
    print(f"  Output dir: {args.output_dir}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/traces", exist_ok=True)

    # Initialize model
    print("Initializing model...")
    print(f"  Loading from: {config.model_name}")
    if enable_world:
        print(f"  Cosmos model: {config.cosmos_model_name}")

    model = TheWorld.from_pretrained(
        config.model_name,
        enable_world=enable_world,
        cosmos_model_name=config.cosmos_model_name,
        freeze_gemma_vision=config.freeze_gemma_vision,
        freeze_gemma_language=config.freeze_gemma_language,
        freeze_cosmos_vae=config.freeze_cosmos_vae,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32,
        device_map="auto",
    )

    if config.use_gradient_checkpointing:
        print("  Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Print trainable parameters
    trainable, total, percentage = model.get_trainable_parameters()
    print()
    print("Model parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({percentage:.4f}%)")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(config, args.num_samples)

    try:
        dataset_size = len(dataset)
        print(f"  Dataset size: {dataset_size:,}")
    except TypeError:
        print(f"  Dataset size: streaming (no length)")

    # Create data collator
    data_collator = create_theworld_collator(model, max_length=config.max_seq_length)
    print(f"  Max sequence length: {config.max_seq_length}")
    print()

    # Create a single batch
    print("Creating batch...")
    batch_items = [dataset[i] for i in range(min(config.batch_size, len(dataset)))]
    batch = data_collator(batch_items)

    # Move batch to GPU
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    print(f"  Batch size: {batch['input_ids'].shape[0]}")
    print(f"  Sequence length: {batch['input_ids'].shape[1]}")
    print(f"  Device: {device}")
    print()

    # Setup optimizer
    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print()

    # Profiling configuration
    print("=" * 80)
    print("Starting Profiling")
    print("=" * 80)
    print(f"  Warmup steps: 1")
    print(f"  Active profiling steps: {args.profile_steps}")
    print(f"  Total iterations: {args.profile_steps + 2}")
    print()
    print("Profiling features:")
    print("  ✓ CPU time and operations")
    print("  ✓ CUDA kernels and memory transfers")
    print("  ✓ Memory allocations (peak, alloc, dealloc)")
    print("  ✓ Stack traces (exact code locations)")
    print("  ✓ Tensor shapes")
    print()

    # Setup profiler
    prof = profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        schedule=schedule(
            wait=0,  # Start immediately
            warmup=1,  # Warmup 1 step
            active=args.profile_steps,  # Profile N steps
            repeat=1,  # Run once
        ),
        on_trace_ready=tensorboard_trace_handler(f"{args.output_dir}/traces"),
        record_shapes=True,  # Record tensor shapes
        profile_memory=True,  # Profile memory allocations
        with_stack=True,  # Capture stack traces
        with_flops=True,  # Estimate FLOPs
    )

    # Run profiled training iterations
    model.train()

    with prof:
        for step in range(args.profile_steps + 2):  # warmup + active steps
            print(f"Step {step + 1}/{args.profile_steps + 2}...", end=" ", flush=True)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                images=batch["images"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            if not args.forward_only:
                # Backward pass
                loss.backward()

                # Optimizer step
                optimizer.step()

            print(f"loss={loss.item():.4f}")

            # Step profiler
            prof.step()

    print()
    print("=" * 80)
    print("Profiling Complete")
    print("=" * 80)
    print()

    # Save profiling data to structured files
    print("=" * 80)
    print("SAVING PROFILING DATA")
    print("=" * 80)
    save_profiling_data(prof, args.output_dir, config, enable_world, top_k=100)

    # Print summary tables (top 30 for console)
    print("=" * 80)
    print("TOP 30 OPERATIONS BY CUDA TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    print()

    print("=" * 80)
    print("TOP 30 OPERATIONS BY CPU TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
    print()

    print("=" * 80)
    print("TOP 30 OPERATIONS BY MEMORY")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
    print()

    # Print memory summary
    print("=" * 80)
    print("GPU MEMORY SUMMARY")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print()

    # Export chrome trace
    print("=" * 80)
    print("PROFILING OUTPUT")
    print("=" * 80)
    print(f"✓ TensorBoard traces: {args.output_dir}/traces/")
    print(f"  View with: tensorboard --logdir {args.output_dir}/traces")
    print()
    print(f"✓ Chrome trace: {args.output_dir}/traces/*.pt.trace.json")
    print(f"  View in Chrome: chrome://tracing")
    print()
    print(f"✓ Structured data files: {args.output_dir}/")
    print(f"  - profiling_summary.json (comprehensive JSON)")
    print(f"  - top_cuda_operations.csv")
    print(f"  - top_cpu_operations.csv")
    print(f"  - top_memory_operations.csv")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
