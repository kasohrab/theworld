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
from transformers.trainer_callback import EarlyStoppingCallback
from huggingface_hub import snapshot_download

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
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
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
    - Custom datasets (dataset_name="custom")

    Args:
        config: TrainingConfig with dataset settings

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    from theworld.datasets import load_datacomp

    # Authenticate with HuggingFace if token provided
    if config.hf_token:
        try:
            import huggingface_hub

            huggingface_hub.login(token=config.hf_token, add_to_git_credential=False)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            print(f"⚠ HuggingFace authentication failed: {e}")

    if config.dataset_name == "datacomp":
        print(f"Loading DataComp-1B dataset...")
        print(f"  Samples: {config.num_samples if config.num_samples else 'all (1.4B)'}")
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
        raise ValueError(f"Unknown dataset_name: {config.dataset_name}. " f"Supported: 'datacomp', 'custom'")

    return train_dataset, eval_dataset


def resolve_checkpoint_path(checkpoint_path: str, hf_token: str = None) -> str:
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

    # Initialize model
    print(f"\nInitializing model...")
    model = TheWorld(
        config.model_name,
        cosmos_model_name=config.cosmos_model_name,
        device=config.device,
        num_world_steps=config.num_world_steps,
        max_world_steps=config.max_world_steps,
        freeze_gemma_vision=config.freeze_gemma_vision,
        freeze_gemma_language=config.freeze_gemma_language,
        freeze_cosmos_vae=config.freeze_cosmos_vae,
    )

    # Enable gradient checkpointing if configured
    if config.use_gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.enable_gradient_checkpointing()

    # Print trainable parameters
    trainable, total, percentage = model.get_trainable_parameters()
    print(f"\nModel parameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,} ({percentage:.4f}%)")

    # Load datasets
    print(f"\nLoading datasets...")
    train_dataset, eval_dataset = load_datasets(config)
    print(f"  Train size: {len(train_dataset)}")
    if eval_dataset:
        print(f"  Eval size: {len(eval_dataset)}")

    # Create data collator
    data_collator = create_theworld_collator(model)

    # Setup training arguments
    print(f"\nSetting up HuggingFace Trainer...")

    # Check if model already uses device_map (e.g., from device_map="auto")
    # If so, we need to prevent Trainer from trying to move it
    has_device_map = hasattr(model, "hf_device_map")
    if has_device_map:
        print("  ⚠ Model uses device_map='auto' - Trainer will skip device placement")
        print(f"  Device map: {model.hf_device_map}")

    # Determine mixed precision settings
    fp16 = config.mixed_precision == "fp16"
    bf16 = config.mixed_precision == "bf16"

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        # Training
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
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
        # Distributed training
        ddp_find_unused_parameters=False if has_device_map else None,
        local_rank=args.local_rank,
    )

    # Setup WandB if configured
    if config.log_to_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )
        training_args.report_to.append("wandb")

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
