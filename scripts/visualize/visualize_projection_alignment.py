"""
Visualize the alignment of Cosmos world embeddings with Gemma embedding space.

This script:
1. Loads a trained TheWorld model
2. Processes a batch of images to extract world embeddings
3. Computes alignment metrics (cosine similarity, L2 distance)
4. Visualizes the distribution using PCA/t-SNE

Usage:
    python scripts/visualize_projection_alignment.py \
        --model checkpoints/theworld-vsr/checkpoint-1000 \
        --dataset vsr \
        --num_samples 100 \
        --output visualizations/projection_alignment.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from theworld import TheWorld
from theworld.datasets import load_vsr, load_datacomp


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize projection alignment")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        default="vsr",
        choices=["vsr", "datacomp"],
        help="Dataset to use for visualization",
    )
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to analyze")
    parser.add_argument("--output", type=str, default="projection_alignment.png", help="Output visualization path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/home/hice1/ksohrab3/scratch/theworld/data/images",
        help="Image folder for VSR dataset",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str) -> TheWorld:
    """Load TheWorld model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Try loading as TheWorld checkpoint first
    try:
        model = TheWorld.from_pretrained(
            "google/gemma-3-4b-it",
            enable_world=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        # Load checkpoint weights
        checkpoint = torch.load(Path(checkpoint_path) / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        print("✓ Loaded model from checkpoint")
    except Exception as e:
        print(f"⚠ Failed to load checkpoint: {e}")
        print("Using pretrained model instead")
        model = TheWorld.from_pretrained(
            "google/gemma-3-4b-it",
            enable_world=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model


def extract_embeddings(model: TheWorld, dataset, num_samples: int, batch_size: int) -> Dict[str, np.ndarray]:
    """Extract world embeddings and projected embeddings from the model.

    Returns:
        Dictionary with:
        - cosmos_embeddings: Raw Cosmos latents (B, 16, H, W)
        - projected_embeddings: Projected to Gemma space (B, 2304, H, W)
        - gemma_vision_embeddings: Gemma vision embeddings for comparison (B, ~256, 2304)
    """
    cosmos_embeddings_list = []
    projected_embeddings_list = []
    gemma_vision_embeddings_list = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in tqdm(range(0, min(num_samples, len(dataset)), batch_size), desc="Extracting embeddings"):
            # Get batch
            batch_end = min(i + batch_size, min(num_samples, len(dataset)))
            batch_items = [dataset[j] for j in range(i, batch_end)]

            # Process images
            images = [item["image"] for item in batch_items]
            texts = [item["text"] for item in batch_items]

            # Prepare inputs
            messages_batch = []
            for img, text in zip(images, texts):
                messages_batch.append(
                    [
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": img}, {"type": "text", "text": text}],
                        }
                    ]
                )

            # Tokenize
            inputs = model.processor.apply_chat_template(
                messages_batch, tokenize=True, return_dict=True, return_tensors="pt"
            )
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Get embeddings from model internals
            # We need to access intermediate representations

            # 1. Get Cosmos world embeddings (before projection)
            if hasattr(model, "cosmos_encoder") and model.cosmos_encoder is not None:
                # Extract cosmos latents
                cosmos_latents = model.cosmos_encoder.encode_images(
                    images, num_world_steps=0
                )  # Shape: (B, 16, 1, H, W)
                cosmos_latents = cosmos_latents.squeeze(2)  # (B, 16, H, W)

                # Get projected embeddings
                B, C, H, W = cosmos_latents.shape
                cosmos_flat = cosmos_latents.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, 16)
                projected = model.cosmos_encoder.projection(cosmos_flat)  # (B, H*W, 2304)

                cosmos_embeddings_list.append(cosmos_latents.cpu().numpy())
                projected_embeddings_list.append(projected.cpu().numpy())

            # 2. Get Gemma vision embeddings
            pixel_values = inputs["pixel_values"]
            vision_features = model.gemma.model.get_image_features(pixel_values)  # (B, ~256, 2304)
            gemma_vision_embeddings_list.append(vision_features.cpu().numpy())

    # Concatenate all batches
    result = {}
    if cosmos_embeddings_list:
        result["cosmos_embeddings"] = np.concatenate(cosmos_embeddings_list, axis=0)
        result["projected_embeddings"] = np.concatenate(projected_embeddings_list, axis=0)
    if gemma_vision_embeddings_list:
        result["gemma_vision_embeddings"] = np.concatenate(gemma_vision_embeddings_list, axis=0)

    return result


def compute_alignment_metrics(embeddings: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute alignment metrics between projected world embeddings and Gemma vision embeddings."""
    metrics = {}

    if "projected_embeddings" not in embeddings or "gemma_vision_embeddings" not in embeddings:
        return metrics

    # Flatten spatial dimensions: (B, H*W, 2304) -> (B*H*W, 2304)
    projected = embeddings["projected_embeddings"]
    B, HW, D = projected.shape
    projected_flat = projected.reshape(-1, D)  # (B*H*W, 2304)

    # Gemma vision: (B, N, 2304) -> (B*N, 2304)
    gemma_vision = embeddings["gemma_vision_embeddings"]
    B_v, N, D = gemma_vision.shape
    gemma_flat = gemma_vision.reshape(-1, D)  # (B*N, 2304)

    # Sample equal number of vectors for fair comparison
    n_samples = min(projected_flat.shape[0], gemma_flat.shape[0], 10000)
    projected_sample = projected_flat[np.random.choice(projected_flat.shape[0], n_samples, replace=False)]
    gemma_sample = gemma_flat[np.random.choice(gemma_flat.shape[0], n_samples, replace=False)]

    # Normalize for cosine similarity
    projected_norm = projected_sample / (np.linalg.norm(projected_sample, axis=1, keepdims=True) + 1e-8)
    gemma_norm = gemma_sample / (np.linalg.norm(gemma_sample, axis=1, keepdims=True) + 1e-8)

    # Compute pairwise cosine similarity (sample-wise)
    cosine_sim = np.sum(projected_norm * gemma_norm, axis=1)
    metrics["mean_cosine_similarity"] = float(np.mean(cosine_sim))
    metrics["std_cosine_similarity"] = float(np.std(cosine_sim))

    # L2 distance
    l2_dist = np.linalg.norm(projected_sample - gemma_sample, axis=1)
    metrics["mean_l2_distance"] = float(np.mean(l2_dist))
    metrics["std_l2_distance"] = float(np.std(l2_dist))

    # Embedding magnitude
    metrics["projected_magnitude_mean"] = float(np.mean(np.linalg.norm(projected_sample, axis=1)))
    metrics["gemma_magnitude_mean"] = float(np.mean(np.linalg.norm(gemma_sample, axis=1)))

    return metrics


def visualize_embeddings(embeddings: Dict[str, np.ndarray], output_path: str, metrics: Dict[str, float]):
    """Create visualization of embedding alignment using PCA and t-SNE."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Prepare data
    projected = embeddings["projected_embeddings"]
    B, HW, D = projected.shape
    projected_flat = projected.reshape(-1, D)

    gemma_vision = embeddings["gemma_vision_embeddings"]
    B_v, N, D = gemma_vision.shape
    gemma_flat = gemma_vision.reshape(-1, D)

    # Sample for visualization (t-SNE is slow on large datasets)
    n_vis = min(1000, projected_flat.shape[0], gemma_flat.shape[0])
    proj_sample = projected_flat[np.random.choice(projected_flat.shape[0], n_vis, replace=False)]
    gemma_sample = gemma_flat[np.random.choice(gemma_flat.shape[0], n_vis, replace=False)]

    # Combine for joint visualization
    combined = np.vstack([proj_sample, gemma_sample])
    labels = np.array([0] * n_vis + [1] * n_vis)  # 0=projected, 1=gemma

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined)

    axes[0, 0].scatter(
        pca_result[labels == 0, 0], pca_result[labels == 0, 1], alpha=0.5, s=10, label="Projected World", c="blue"
    )
    axes[0, 0].scatter(
        pca_result[labels == 1, 0], pca_result[labels == 1, 1], alpha=0.5, s=10, label="Gemma Vision", c="red"
    )
    axes[0, 0].set_title("PCA: Embedding Space Alignment")
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # t-SNE
    print("Computing t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
    tsne_result = tsne.fit_transform(combined)

    axes[0, 1].scatter(
        tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], alpha=0.5, s=10, label="Projected World", c="blue"
    )
    axes[0, 1].scatter(
        tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], alpha=0.5, s=10, label="Gemma Vision", c="red"
    )
    axes[0, 1].set_title("t-SNE: Embedding Space Alignment")
    axes[0, 1].set_xlabel("t-SNE 1")
    axes[0, 1].set_ylabel("t-SNE 2")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Cosine similarity distribution
    projected_norm = proj_sample / (np.linalg.norm(proj_sample, axis=1, keepdims=True) + 1e-8)
    gemma_norm = gemma_sample / (np.linalg.norm(gemma_sample, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.sum(projected_norm * gemma_norm, axis=1)

    axes[1, 0].hist(cosine_sim, bins=50, alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(metrics["mean_cosine_similarity"], color="red", linestyle="--", label="Mean")
    axes[1, 0].set_title("Cosine Similarity Distribution")
    axes[1, 0].set_xlabel("Cosine Similarity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Metrics summary
    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment="center", family="monospace")
    axes[1, 1].set_title("Alignment Metrics")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to {output_path}")


def main():
    args = parse_args()

    # Load model
    model = load_model(args.model)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "vsr":
        dataset = load_vsr(
            split="validation", variant="random", image_folder=args.image_folder, num_samples=args.num_samples
        )
    else:  # datacomp
        dataset = load_datacomp(split="train", num_samples=args.num_samples, streaming=False)

    # Extract embeddings
    print(f"Extracting embeddings from {args.num_samples} samples...")
    embeddings = extract_embeddings(model, dataset, args.num_samples, args.batch_size)

    # Compute metrics
    print("Computing alignment metrics...")
    metrics = compute_alignment_metrics(embeddings)

    print("\nAlignment Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Create visualization
    print("Creating visualization...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_embeddings(embeddings, str(output_path), metrics)

    print(f"\n✓ Done! Visualization saved to {output_path}")


if __name__ == "__main__":
    main()
