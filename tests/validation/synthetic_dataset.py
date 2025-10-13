"""
Synthetic dataset for fast testing without network dependencies.

Generates synthetic images on-the-fly for quick validation of training pipelines.
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    Synthetic dataset that generates random images on-the-fly.

    No network dependencies, no disk I/O - perfect for quick smoke tests.

    Example:
        >>> dataset = SyntheticDataset(num_samples=10)
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['image', 'text', 'label'])
    """

    def __init__(
        self,
        num_samples: int = 10,
        image_size: int = 224,
        question_template: str = "Describe this image in detail.",
        label_template: str = "This is a synthetic test image with random colors.",
    ):
        """
        Initialize synthetic dataset.

        Args:
            num_samples: Number of samples in the dataset
            image_size: Size of generated square images
            question_template: Question text for each sample
            label_template: Label/answer text for each sample
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.question_template = question_template
        self.label_template = label_template

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a synthetic sample.

        Returns:
            dict with keys: 'image' (PIL Image), 'text' (str), 'label' (str)
        """
        # Generate random RGB image
        # Use idx as seed for reproducibility
        rng = np.random.RandomState(idx)
        img_array = rng.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        image = Image.fromarray(img_array, mode="RGB")

        return {
            "image": image,
            "text": self.question_template,
            "label": self.label_template,
        }
