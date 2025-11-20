"""Simple training test: train → save → resume with projection architecture config.

This test creates dummy data to verify:
1. Training works with the new projection_architecture config
2. Checkpoint saving preserves architecture settings
3. Resuming from checkpoint maintains all settings
4. Training continues correctly after resume
"""

import os
import sys
import tempfile
import torch
from PIL import Image
import numpy as np
from transformers import TrainingArguments, Trainer

# Add python/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../..", "python"))

from theworld import TheWorld
from theworld.data import create_theworld_collator, TheWorldDataset


def create_dummy_data(num_samples=10):
    """Create dummy dataset for testing."""
    data = []
    for i in range(num_samples):
        # Create a random RGB image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        data.append({
            "image": img,
            "text": f"Question {i}: What is in this image?",
            "label": f"Answer {i}: This is a test image."
        })
    return data


def main():
    print("=" * 70)
    print("Simple Training Test with Projection Architecture Config")
    print("=" * 70)
    print()

    # Configuration
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN not set - will use cached models only")
    print()

    # Test parameters
    NUM_TRAIN_SAMPLES = 10
    STEPS_BEFORE_SAVE = 2
    STEPS_AFTER_RESUME = 2

    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "test_checkpoint")

        print("=" * 70)
        print("Phase 1: Initial Training (2 steps)")
        print("=" * 70)
        print()

        # Load model with channel mode and mlp architecture
        print("Loading TheWorld model...")
        print("  - projection_architecture: mlp")
        print("  - world_projection_mode: channel")
        print()

        model = TheWorld.from_pretrained(
            "google/gemma-3-4b-it",
            enable_world=True,
            world_projection_mode="channel",
            projection_architecture="mlp",
            freeze_gemma_vision=True,
            freeze_gemma_language=True,
            freeze_cosmos_vae=True,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        # Verify config
        assert model.config.projection_architecture == "mlp"
        assert model.config.world_projection_mode == "channel"
        print("✓ Model loaded with correct settings")

        trainable, total, pct = model.get_trainable_parameters()
        print(f"✓ Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
        print()

        # Create dummy dataset
        print(f"Creating dummy dataset ({NUM_TRAIN_SAMPLES} samples)...")
        dummy_data = create_dummy_data(NUM_TRAIN_SAMPLES)
        train_dataset = TheWorldDataset(dummy_data)
        print(f"✓ Dataset created: {len(train_dataset)} samples")
        print()

        # Setup training
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=STEPS_BEFORE_SAVE,  # Save after 2 steps
            save_total_limit=1,
            logging_steps=1,
            learning_rate=1e-4,
            bf16=True,
            max_steps=STEPS_BEFORE_SAVE,  # Stop after 2 steps
            report_to=[],  # Disable wandb/tensorboard
            remove_unused_columns=False,
        )

        collator = create_theworld_collator(model=model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collator,
        )

        print(f"Training for {STEPS_BEFORE_SAVE} steps...")
        print()
        trainer.train()
        print()
        print(f"✅ Phase 1 complete: Trained for {STEPS_BEFORE_SAVE} steps")
        print(f"✓ Checkpoint saved to: {checkpoint_dir}")
        print()

        # Verify checkpoint exists
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{STEPS_BEFORE_SAVE}")
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        print(f"✓ Checkpoint verified: {checkpoint_path}")
        print()

        # Clean up first model
        del model
        del trainer
        torch.cuda.empty_cache()

        print("=" * 70)
        print("Phase 2: Resume Training (2 more steps)")
        print("=" * 70)
        print()

        # Load from checkpoint
        print(f"Loading from checkpoint: {checkpoint_path}")
        model2 = TheWorld.from_checkpoint(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto"
        )

        # Verify settings preserved
        print("Verifying checkpoint settings...")
        assert model2.config.projection_architecture == "mlp", \
            f"Architecture mismatch: expected mlp, got {model2.config.projection_architecture}"
        assert model2.config.world_projection_mode == "channel", \
            f"Mode mismatch: expected channel, got {model2.config.world_projection_mode}"
        print("✓ projection_architecture: mlp")
        print("✓ world_projection_mode: channel")

        trainable2, total2, pct2 = model2.get_trainable_parameters()
        assert trainable2 == trainable, f"Trainable params changed: {trainable} -> {trainable2}"
        print(f"✓ Trainable params unchanged: {trainable2:,} ({pct2:.2f}%)")
        print()

        # Resume training
        training_args2 = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            save_steps=100,  # Don't save again
            logging_steps=1,
            learning_rate=1e-4,
            bf16=True,
            max_steps=STEPS_BEFORE_SAVE + STEPS_AFTER_RESUME,  # Train 2 more steps (total 4)
            report_to=[],
            remove_unused_columns=False,
        )

        trainer2 = Trainer(
            model=model2,
            args=training_args2,
            train_dataset=train_dataset,
            data_collator=collator,
        )

        print(f"Resuming training for {STEPS_AFTER_RESUME} more steps...")
        print()
        trainer2.train(resume_from_checkpoint=checkpoint_path)
        print()
        print(f"✅ Phase 2 complete: Trained for {STEPS_AFTER_RESUME} more steps")
        print()

        print("=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print(f"  ✓ Initial training: {STEPS_BEFORE_SAVE} steps")
        print(f"  ✓ Checkpoint saved with projection_architecture=mlp, mode=channel")
        print(f"  ✓ Resumed from checkpoint with settings preserved")
        print(f"  ✓ Continued training: {STEPS_AFTER_RESUME} more steps")
        print(f"  ✓ Total training steps: {STEPS_BEFORE_SAVE + STEPS_AFTER_RESUME}")
        print()


if __name__ == "__main__":
    main()
