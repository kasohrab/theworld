"""
Training configuration for TheWorld model.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training TheWorld model with HuggingFace Trainer.

    Model Configuration:
        model_name: HuggingFace model ID for Gemma 3
        num_world_steps: Number of future frames to predict (0 = current only)
        max_world_steps: Maximum frames for temporal embeddings
        freeze_gemma_vision: If True, freeze Gemma's SigLIP vision encoder
        freeze_gemma_language: If True, freeze Gemma's language model
        freeze_cosmos_vae: If True, freeze Cosmos VAE encoder

    Training Hyperparameters:
        learning_rate: Learning rate for optimizer
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping

    Memory Optimization:
        use_gradient_checkpointing: Enable gradient checkpointing to save memory
        mixed_precision: Mixed precision training ("no", "fp16", "bf16")

    Checkpointing:
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps
        save_total_limit: Keep only last N checkpoints (None = keep all)
        resume_from_checkpoint: Path to checkpoint to resume from

    Evaluation:
        eval_steps: Run evaluation every N steps
        eval_batch_size: Batch size for evaluation (None = use batch_size)
        do_eval: Whether to run evaluation

    Logging:
        logging_steps: Log metrics every N steps
        log_to_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        log_to_tensorboard: Whether to log to TensorBoard

    Data:
        max_seq_length: Maximum sequence length for text
        num_workers: Number of dataloader workers
        train_dataset_path: Path to training dataset (HF dataset or custom)
        eval_dataset_path: Path to evaluation dataset
        dataset_name: Name of dataset to use ("datacomp", "custom", etc.)
        num_samples: Limit dataset to N samples (None = use all)
        streaming: Use streaming mode for large datasets
        question_template: Question template for image captioning datasets

    HuggingFace:
        hf_token: HuggingFace API token for private datasets/models
        push_to_hub: Upload checkpoints to HuggingFace Hub
        hub_model_id: Repository name on Hub (e.g., "username/model-name")
        hub_strategy: Upload strategy ("end", "every_save", "checkpoint")
        hub_private_repo: Create private repository on Hub
    """

    # Model configuration
    model_name: str = "google/gemma-3-4b-it"
    cosmos_model_name: str = "nvidia/Cosmos-Predict2-2B-Video2World"
    num_world_steps: int = 0
    max_world_steps: int = 16
    freeze_gemma_vision: bool = True
    freeze_gemma_language: bool = True
    freeze_cosmos_vae: bool = True

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Memory optimization
    use_gradient_checkpointing: bool = False
    mixed_precision: str = "bf16"  # "no", "fp16", "bf16"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: int = 500
    save_total_limit: Optional[int] = 3  # Keep only last 3 checkpoints
    resume_from_checkpoint: Optional[str] = None

    # Evaluation
    eval_steps: int = 500
    eval_batch_size: Optional[int] = None  # Use batch_size if None
    do_eval: bool = False

    # Logging
    logging_steps: int = 10
    log_to_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_to_tensorboard: bool = True

    # Data
    max_seq_length: int = 2048
    num_workers: int = 4
    train_dataset_path: Optional[str] = None
    eval_dataset_path: Optional[str] = None
    dataset_name: str = "custom"  # "datacomp", "custom", etc.
    num_samples: Optional[int] = None  # Limit to N samples (None = all)
    streaming: bool = False  # Use streaming mode for large datasets
    question_template: str = "Describe this image in detail."

    # HuggingFace
    hf_token: Optional[str] = None  # HF API token
    push_to_hub: bool = False  # Upload checkpoints to Hub
    hub_model_id: Optional[str] = None  # e.g., "username/theworld-datacomp"
    hub_strategy: str = "every_save"  # "end", "every_save", "checkpoint"
    hub_private_repo: bool = False  # Create private repository

    # Device
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.mixed_precision not in ["no", "fp16", "bf16"]:
            raise ValueError(f"mixed_precision must be 'no', 'fp16', or 'bf16', got '{self.mixed_precision}'")

        if self.eval_batch_size is None:
            self.eval_batch_size = self.batch_size

        if self.log_to_wandb and self.wandb_project is None:
            raise ValueError("wandb_project must be set when log_to_wandb=True")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create TrainingConfig from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            TrainingConfig instance
        """
        # Filter out comment fields (keys starting with _)
        filtered_dict = {k: v for k, v in config_dict.items() if not k.startswith("_")}
        return cls(**filtered_dict)

    def to_dict(self) -> dict:
        """Convert TrainingConfig to dictionary.

        Returns:
            Dictionary with configuration values
        """
        return {k: v for k, v in self.__dict__.items()}
