"""
HuggingFace Trainer smoke test for refactored TheWorld model.

Tests that the model works end-to-end with HF Trainer in both modes:
1. enable_world=True (with Cosmos world model)
2. enable_world=False (Gemma baseline)
"""

import sys
from pathlib import Path
import torch
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld, create_theworld_collator
from tests.validation.synthetic_dataset import SyntheticDataset
from transformers import Trainer, TrainingArguments


def test_hf_trainer_with_world():
    """Test HF Trainer with enable_world=True (Cosmos world model)."""
    print("=" * 80)
    print("TEST 1: HF Trainer with enable_world=True (Cosmos World Model)")
    print("=" * 80)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_world"

        # Initialize model with world model enabled
        print("\nLoading model with enable_world=True...")
        model = TheWorld.from_pretrained(
            "google/gemma-3-4b-it",
            enable_world=True,
            device="cuda",
            freeze_gemma_vision=True,
            freeze_gemma_language=True,
            freeze_cosmos_vae=True,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        trainable, total, pct = model.get_trainable_parameters()
        print(f"✓ Model loaded: {trainable:,} / {total:,} trainable ({pct:.2f}%)")

        # Create synthetic dataset (no network dependencies)
        print("\nCreating synthetic dataset...")
        train_dataset = SyntheticDataset(num_samples=4)
        print(f"✓ Dataset created: {len(train_dataset)} samples")

        # Create collator
        collate_fn = create_theworld_collator(model)

        # Setup training arguments (minimal config for smoke test)
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,  # Only 2 steps for smoke test
            save_steps=1,  # Save after each step
            save_total_limit=2,
            logging_steps=1,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            bf16=True,
            report_to=[],  # Disable all reporting
        )

        # Create Trainer
        print("\nInitializing HF Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )
        print("✓ Trainer initialized")

        # Train for 2 steps
        print("\nStarting training (2 steps)...")
        trainer.train()
        print("✓ Training completed")

        # Verify checkpoints saved
        checkpoints = list(output_dir.glob("checkpoint-*"))
        print(f"✓ Checkpoints saved: {len(checkpoints)} checkpoints")
        for ckpt in checkpoints:
            ckpt_file = ckpt / "pytorch_model.bin"
            if ckpt_file.exists():
                size_mb = ckpt_file.stat().st_size / 1e6
                print(f"  - {ckpt.name}: {size_mb:.1f} MB")

        # Test loading from checkpoint
        if checkpoints:
            print("\nTesting checkpoint loading...")
            latest_ckpt = sorted(checkpoints)[-1]
            trainer_resumed = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=collate_fn,
            )
            trainer_resumed.train(resume_from_checkpoint=str(latest_ckpt))
            print(f"✓ Successfully resumed from {latest_ckpt.name}")

    print("\n✓ TEST 1 PASSED: HF Trainer works with enable_world=True")


def test_hf_trainer_baseline():
    """Test HF Trainer with enable_world=False (Gemma baseline)."""
    print("\n" + "=" * 80)
    print("TEST 2: HF Trainer with enable_world=False (Gemma Baseline)")
    print("=" * 80)

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "test_baseline"

        # Initialize model WITHOUT world model (Gemma baseline)
        print("\nLoading model with enable_world=False...")
        model = TheWorld.from_pretrained(
            "google/gemma-3-4b-it",
            enable_world=False,  # Gemma-only baseline
            device="cuda",
            freeze_gemma_vision=True,
            freeze_gemma_language=True,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        trainable, total, pct = model.get_trainable_parameters()
        print(f"✓ Model loaded: {trainable:,} / {total:,} trainable ({pct:.2f}%)")

        # Create synthetic dataset
        print("\nCreating synthetic dataset...")
        train_dataset = SyntheticDataset(num_samples=4)
        print(f"✓ Dataset created: {len(train_dataset)} samples")

        # Create collator
        collate_fn = create_theworld_collator(model)

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=2,  # Only 2 steps
            save_steps=1,
            save_total_limit=2,
            logging_steps=1,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            bf16=True,
            report_to=[],
        )

        # Create Trainer
        print("\nInitializing HF Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=collate_fn,
        )
        print("✓ Trainer initialized")

        # Train for 2 steps
        print("\nStarting training (2 steps)...")
        trainer.train()
        print("✓ Training completed")

        # Verify checkpoints saved
        checkpoints = list(output_dir.glob("checkpoint-*"))
        print(f"✓ Checkpoints saved: {len(checkpoints)} checkpoints")
        for ckpt in checkpoints:
            ckpt_file = ckpt / "pytorch_model.bin"
            if ckpt_file.exists():
                size_mb = ckpt_file.stat().st_size / 1e6
                print(f"  - {ckpt.name}: {size_mb:.1f} MB")

    print("\n✓ TEST 2 PASSED: HF Trainer works with enable_world=False")


def main():
    """Run both smoke tests."""
    print("\n" + "=" * 80)
    print("HUGGINGFACE TRAINER SMOKE TEST - REFACTORED THEWORLD")
    print("=" * 80)

    try:
        # Test 1: With world model
        test_hf_trainer_with_world()

        # Test 2: Baseline (no world model)
        test_hf_trainer_baseline()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe refactored TheWorld model works correctly with HuggingFace Trainer!")
        print("Both enable_world=True and enable_world=False modes are functional.")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
