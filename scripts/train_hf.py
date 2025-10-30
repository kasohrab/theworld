"""
Training script for TheWorld model using HuggingFace Trainer.

This script provides a production-ready training pipeline with:
- Automatic checkpointing and resuming
- Distributed training support (DDP, FSDP)
- Mixed precision training
- Logging to TensorBoard/WandB
- Gradient accumulation and checkpointing
- HuggingFace Hub upload and download support

Example usage:
    # Train with default config
    python scripts/train_hf.py

    # Train with custom config
    python scripts/train_hf.py --config configs/custom.json

    # Resume from local checkpoint
    python scripts/train_hf.py --resume_from checkpoints/checkpoint-1000

    # Resume from HuggingFace Hub checkpoint
    python scripts/train_hf.py --resume_from username/theworld-datacomp
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments
from huggingface_hub import snapshot_download
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld, TrainingConfig, TheWorldDataset, create_theworld_collator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TheWorld model with HuggingFace Trainer")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to training config JSON file",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (datacomp, custom, etc.)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit dataset to N samples",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets",
    )

    return parser.parse_args()


def load_config(config_path: str) -> TrainingConfig:
    """Load training configuration from JSON file."""
    if not os.path.exists(config_path):
        print(f"⚠ Config file not found: {config_path}")
        print("Using default configuration")
        return TrainingConfig()

    with open(config_path, "r") as f:
        config_dict = json.load(f)

    return TrainingConfig.from_dict(config_dict)


def load_datasets(config: TrainingConfig):
    """Load training and evaluation datasets.

    Supports:
    - DataComp-1B (dataset_name="datacomp")
    - VSR (dataset_name="vsr")
    - LLaVA-CC3M-Pretrain-595K (dataset_name="llava_pretrain")
    - Custom datasets (dataset_name="custom")

    Args:
        config: TrainingConfig with dataset settings

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from theworld.datasets import load_datacomp, load_vsr, load_llava_pretrain

    # Authenticate with HuggingFace if token provided
    if config.hf_token:
        try:
            import huggingface_hub

            huggingface_hub.login(token=config.hf_token, add_to_git_credential=False)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            print(f"⚠ HuggingFace authentication failed: {e}")

    if config.dataset_name == "datacomp":
        print(f"Loading DataComp-Small dataset...")
        print(f"  Samples: {config.num_samples if config.num_samples else 'all (12.8M)'}")
        print(f"  Streaming: {config.streaming}")

        train_dataset = load_datacomp(
            split="train",
            num_samples=config.num_samples,
            streaming=config.streaming,
            question_template=config.question_template,
            hf_token=config.hf_token,
        )

        # DataComp doesn't have a separate eval split, use subset of train
        eval_dataset = None
        if config.do_eval:
            eval_samples = min(1000, config.num_samples // 10) if config.num_samples else 1000
            eval_dataset = load_datacomp(
                split="train",
                num_samples=eval_samples,
                streaming=False,  # Eval should not be streaming
                question_template=config.question_template,
                hf_token=config.hf_token,
            )

    elif config.dataset_name == "vsr":
        print(f"Loading VSR dataset...")
        print(f"  Samples: {config.num_samples if config.num_samples else 'all (~10K)'}")

        # Get image folder from config (or use default)
        image_folder = getattr(config, "image_folder", None)
        if image_folder is None:
            image_folder = "/home/hice1/ksohrab3/scratch/theworld/data/images"

        train_dataset = load_vsr(
            split="train",
            variant="random",
            image_folder=image_folder,
            num_samples=config.num_samples,
            question_template=config.question_template,
            hf_token=config.hf_token,
        )

        # Load validation split for evaluation
        eval_dataset = None
        if config.do_eval:
            eval_dataset = load_vsr(
                split="validation",
                variant="random",
                image_folder=image_folder,
                num_samples=None,  # Use full validation set
                question_template=config.question_template,
                hf_token=config.hf_token,
            )

    elif config.dataset_name == "llava_pretrain":
        print(f"Loading LLaVA-CC3M-Pretrain-595K dataset...")
        print(f"  Samples: {config.num_samples if config.num_samples else 'all (595K)'}")

        # Get image folder from config (or use default)
        image_folder = getattr(config, "image_folder", None)
        if image_folder is None:
            image_folder = "data/llava-cc3m/images"

        # Load full dataset
        full_dataset = load_llava_pretrain(
            image_folder=image_folder,
            num_samples=config.num_samples,
            hf_token=config.hf_token,
            auto_download=True,
        )

        # Split into train/val (hold out 1% for validation)
        eval_dataset = None
        if config.do_eval:
            from torch.utils.data import random_split

            total_size = len(full_dataset)
            val_size = max(100, int(0.01 * total_size))  # At least 100 samples, or 1% of dataset
            train_size = total_size - val_size

            train_dataset, eval_dataset = random_split(full_dataset, [train_size, val_size])
            print(f"  Split: {train_size:,} train, {val_size:,} validation")
        else:
            train_dataset = full_dataset

    elif config.dataset_name == "custom":
        # Custom dataset loading
        # Users can modify this section for their own datasets
        print("⚠ Using dummy custom dataset. Modify load_datasets() for your data.")

        dummy_image = torch.randn(3, 896, 896)
        train_data = [
            {"image": dummy_image, "text": f"Question {i}?", "label": f"Answer {i}"}
            for i in range(config.num_samples if config.num_samples else 100)
        ]
        train_dataset = TheWorldDataset(train_data)

        eval_dataset = None
        if config.do_eval:
            eval_data = [{"image": dummy_image, "text": f"Question {i}?", "label": f"Answer {i}"} for i in range(20)]
            eval_dataset = TheWorldDataset(eval_data)

    else:
        raise ValueError(
            f"Unknown dataset_name: {config.dataset_name}. " f"Supported: 'datacomp', 'vsr', 'llava_pretrain', 'custom'"
        )

    return train_dataset, eval_dataset


def resolve_checkpoint_path(checkpoint_path: str, hf_token: Optional[str] = None) -> Optional[str]:
    """Resolve checkpoint path, downloading from Hub if needed.

    Args:
        checkpoint_path: Local path or HuggingFace Hub model ID
        hf_token: HuggingFace API token for private models

    Returns:
        Local path to checkpoint directory

    Example:
        >>> path = resolve_checkpoint_path("username/theworld-model")
        >>> # Downloads from Hub and returns local path
    """
    if checkpoint_path is None:
        return None

    # Check if it's a Hub model ID (contains "/" and doesn't exist locally)
    is_hub_id = "/" in checkpoint_path and not os.path.exists(checkpoint_path)

    if is_hub_id:
        print(f"\nDownloading checkpoint from HuggingFace Hub: {checkpoint_path}")

        try:
            # Download entire checkpoint directory from Hub
            local_path = snapshot_download(
                repo_id=checkpoint_path,
                token=hf_token,
                allow_patterns=["checkpoint-*/**", "pytorch_model.bin", "*.json", "*.txt"],
            )

            print(f"✓ Checkpoint downloaded to: {local_path}")

            # Find the most recent checkpoint subdirectory if it exists
            checkpoint_dirs = [d for d in os.listdir(local_path) if d.startswith("checkpoint-")]

            if checkpoint_dirs:
                # Sort by checkpoint number and get the latest
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = os.path.join(local_path, checkpoint_dirs[-1])
                print(f"Using latest checkpoint: {checkpoint_dirs[-1]}")
                return latest_checkpoint
            else:
                return local_path

        except Exception as e:
            print(f"⚠ Failed to download from Hub: {e}")
            print(f"Trying to use as local path: {checkpoint_path}")
            return checkpoint_path

    else:
        # Local path
        return checkpoint_path


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    print("=" * 60)
    print("TheWorld Model - HuggingFace Trainer Training")
    print("=" * 60)

    config = load_config(args.config)

    # Override config with command line args
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.resume_from:
        config.resume_from_checkpoint = args.resume_from
    if args.hf_token:
        config.hf_token = args.hf_token
    if args.dataset:
        config.dataset_name = args.dataset
    if args.num_samples is not None:
        config.num_samples = args.num_samples
    if args.streaming:
        config.streaming = args.streaming

    # Get HF token from environment if not provided
    if not config.hf_token:
        import os

        config.hf_token = os.environ.get("HF_TOKEN")

    print(f"\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_name}")
    if config.num_samples:
        print(f"  Samples: {config.num_samples}")
    print(f"  Streaming: {config.streaming}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    print(f"  Mixed precision: {config.mixed_precision}")
    print(f"  Save format: {'safetensors' if config.save_safetensors else 'pickle'}")

    # Initialize model
    print(f"\nInitializing model...")
    print(f"  Using from_pretrained() for proper initialization...")
    print(f"  World embeddings: enabled")
    print(f"  Cosmos model: {config.cosmos_model_name}")
    print(f"  num_world_steps: {config.num_world_steps}")
    print(f"  Accelerate will handle device placement automatically")

    model = TheWorld.from_pretrained(
        config.model_name,
        enable_world=True,
        cosmos_model_name=config.cosmos_model_name,
        freeze_gemma_vision=config.freeze_gemma_vision,
        freeze_gemma_language=config.freeze_gemma_language,
        freeze_cosmos_vae=config.freeze_cosmos_vae,
        torch_dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float32,
        # No device_map - let Accelerate handle device placement
    )

    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Print trainable parameters
    trainable, total, percentage = model.get_trainable_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({percentage:.4f}%)")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset, eval_dataset = load_datasets(config)

    # Handle streaming datasets (no length)
    try:
        train_size = len(train_dataset)
        print(f"  Train size: {train_size:,}")
    except TypeError:
        print(f"  Train size: streaming (no length)")

    if eval_dataset:
        try:
            eval_size = len(eval_dataset)
            print(f"  Eval size: {eval_size:,}")
        except TypeError:
            print(f"  Eval size: streaming (no length)")

    # Create data collator
    data_collator = create_theworld_collator(model)

    # Setup training arguments
    print(f"\nSetting up HuggingFace Trainer...")

    # Determine mixed precision settings
    fp16 = config.mixed_precision == "fp16"
    bf16 = config.mixed_precision == "bf16"

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        # Training
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps if config.max_steps is not None else -1,  # -1 = use num_epochs
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size if config.eval_batch_size is not None else config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        warmup_steps=config.warmup_steps,
        # Mixed precision
        fp16=fp16,
        bf16=bf16,
        # Checkpointing
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_safetensors=config.save_safetensors,
        # Evaluation
        eval_strategy="steps" if config.do_eval else "no",
        eval_steps=config.eval_steps if config.do_eval else None,
        # Logging
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        report_to=["tensorboard"] if config.log_to_tensorboard else [],
        # HuggingFace Hub
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy=config.hub_strategy,
        hub_private_repo=config.hub_private_repo,
        hub_token=config.hf_token,
        # Other
        dataloader_num_workers=config.num_workers,
        remove_unused_columns=False,  # Important: don't remove our custom columns
        # Distributed training - needed because we freeze most parameters
        ddp_find_unused_parameters=True,
    )

    # Setup WandB if configured
    if config.log_to_wandb:
        import wandb

        try:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.to_dict(),
            )
            # Ensure report_to is a list before appending
            if isinstance(training_args.report_to, list):
                training_args.report_to.append("wandb")
            else:
                training_args.report_to = ["wandb"]
            print("✓ Wandb logging enabled")
        except Exception as e:
            print(f"⚠ Wandb initialization failed: {e}")
            print("  Continuing with TensorBoard logging only")
            print("  To enable wandb: Run 'wandb login' or set WANDB_API_KEY environment variable")

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Resume from checkpoint if specified (supports both local paths and Hub model IDs)
    resume_checkpoint = None
    if config.resume_from_checkpoint:
        resume_checkpoint = resolve_checkpoint_path(config.resume_from_checkpoint, hf_token=config.hf_token)

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"⚠ Checkpoint not found: {config.resume_from_checkpoint}")
            resume_checkpoint = None

    # Train!
    print(f"\nStarting training...")
    print("=" * 60)

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print(f"\nSaving final model to {config.output_dir}/final")
    trainer.save_model(os.path.join(config.output_dir, "final"))

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    if config.log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
