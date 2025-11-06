"""Utility functions for drawing bounding boxes on images for spatial reasoning tasks."""

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_bounding_boxes(
    image: Image.Image,
    bboxes: List[List[float]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    thickness: int = 3,
) -> Image.Image:
    """Draw bounding boxes with labels on an image.

    Args:
        image: PIL Image to draw on
        bboxes: List of bounding boxes as [x1, y1, x2, y2] (pixel coordinates)
        labels: Optional labels for each box (e.g., ["Region 0", "Region 1"])
        colors: Optional colors for each box (default: use predefined palette)
        thickness: Line thickness in pixels

    Returns:
        New PIL Image with bounding boxes drawn

    Example:
        >>> img = Image.open("image.jpg")
        >>> bboxes = [[100, 100, 200, 200], [300, 150, 400, 250]]
        >>> labeled_img = draw_bounding_boxes(img, bboxes, labels=["Region 0", "Region 1"])
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Default color palette (high contrast colors)
    default_colors = [
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
        "#FF00FF",  # Magenta
        "#00FFFF",  # Cyan
        "#FF8000",  # Orange
        "#8000FF",  # Purple
    ]

    if colors is None:
        colors = [default_colors[i % len(default_colors)] for i in range(len(bboxes))]

    if labels is None:
        labels = [f"Region {i}" for i in range(len(bboxes))]

    # Try to load a font (fallback to default if not available)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[i]
        label = labels[i]

        # Draw rectangle with thick outline
        for offset in range(thickness):
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=color,
                width=1,
            )

        # Draw label background
        label_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(label_bbox, fill=color)

        # Draw label text
        draw.text((x1, y1 - 20), label, fill="white", font=font)

    return img_copy


def clamp_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """Clamp bounding box coordinates to image boundaries and normalize.

    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Clamped and normalized bounding box (ensures x1 <= x2, y1 <= y2, and minimum 1px area)

    Example:
        >>> clamp_bbox([-10, -5, 200, 300], 100, 100)
        [0, 0, 100, 100]
        >>> clamp_bbox([50, 80, 30, 20], 100, 100)  # Inverted bbox
        [30, 20, 50, 80]
    """
    x1, y1, x2, y2 = bbox

    # Clamp to image boundaries
    x1 = max(0, min(image_width, x1))
    x2 = max(0, min(image_width, x2))
    y1 = max(0, min(image_height, y1))
    y2 = max(0, min(image_height, y2))

    # Normalize coordinates (fix inverted bboxes)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    # Ensure minimum 1px area (prevent zero-area boxes)
    if x2 == x1:
        x2 = min(x1 + 1, image_width)
    if y2 == y1:
        y2 = min(y1 + 1, image_height)

    return [x1, y1, x2, y2]


def decode_rle_mask(rle: dict, image_width: int, image_height: int) -> np.ndarray:
    """Decode RLE mask to binary numpy array.

    Args:
        rle: COCO RLE format dict with 'counts' and 'size'
        image_width: Image width
        image_height: Image height

    Returns:
        Binary mask as numpy array (H, W) with 0/1 values

    Example:
        >>> rle = {'size': [100, 100], 'counts': '...'}
        >>> mask = decode_rle_mask(rle, 100, 100)
        >>> mask.shape
        (100, 100)
    """
    try:
        from pycocotools import mask as cocomask

        mask = cocomask.decode(rle)
        return mask.astype(np.uint8)
    except ImportError:
        raise ImportError("pycocotools is required for RLE mask decoding. " "Install with: pip install pycocotools")


def convert_mask_to_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """Convert binary mask to bounding box.

    Args:
        mask: Binary numpy array (H, W)

    Returns:
        Bounding box as [x1, y1, x2, y2] or None if mask is empty

    Example:
        >>> mask = np.zeros((100, 100))
        >>> mask[20:40, 30:60] = 1
        >>> bbox = convert_mask_to_bbox(mask)
        >>> bbox
        [30, 20, 60, 40]
    """
    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    return [float(x1), float(y1), float(x2), float(y2)]
