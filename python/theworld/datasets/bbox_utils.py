"""Utility functions for drawing bounding boxes on images for spatial reasoning tasks."""

from typing import List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def _labels_overlap(label_bbox1: Tuple[int, int, int, int], label_bbox2: Tuple[int, int, int, int]) -> bool:
    """Check if two label bounding boxes overlap.

    Args:
        label_bbox1: First label bbox as (x1, y1, x2, y2)
        label_bbox2: Second label bbox as (x1, y1, x2, y2)

    Returns:
        True if labels overlap, False otherwise
    """
    x1_1, y1_1, x2_1, y2_1 = label_bbox1
    x1_2, y1_2, x2_2, y2_2 = label_bbox2

    # Check for non-overlap conditions
    if x2_1 <= x1_2 or x2_2 <= x1_1 or y2_1 <= y1_2 or y2_2 <= y1_1:
        return False
    return True


def _find_non_overlapping_positions(
    bboxes: List[List[float]],
    labels: List[str],
    image_size: Tuple[int, int],
    font: ImageFont.FreeTypeFont,
) -> List[Tuple[int, int]]:
    """Find non-overlapping label positions for all bboxes.

    Args:
        bboxes: List of bounding boxes as [x1, y1, x2, y2]
        labels: List of label strings
        image_size: Image dimensions as (width, height)
        font: Font to use for measuring label size

    Returns:
        List of (x, y) positions for each label
    """
    img_width, img_height = image_size
    label_positions_bboxes = []  # Track label bboxes to detect overlaps
    label_positions_xy = []  # Final (x, y) positions

    # Create temporary draw object for measuring text
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    # Position strategies to try (in priority order)
    strategies = [
        "top-outside",
        "bottom-outside",
        "top-inside",
        "right-outside",
        "left-outside",
        "bottom-inside",
    ]

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        x1, y1, x2, y2 = map(int, bbox)

        # Calculate label size
        temp_bbox = temp_draw.textbbox((0, 0), label, font=font)
        label_width = temp_bbox[2] - temp_bbox[0]
        label_height = temp_bbox[3] - temp_bbox[1]
        padding = 2

        # Try each strategy
        found_position = False
        for strategy in strategies:
            # Calculate position based on strategy
            if strategy == "top-outside":
                pos_x = x1
                pos_y = max(0, y1 - label_height - padding)
            elif strategy == "bottom-outside":
                pos_x = x1
                pos_y = min(img_height - label_height, y2 + padding)
            elif strategy == "top-inside":
                pos_x = x1 + padding
                pos_y = y1 + padding
            elif strategy == "bottom-inside":
                pos_x = x1 + padding
                pos_y = max(y1, y2 - label_height - padding)
            elif strategy == "right-outside":
                pos_x = min(img_width - label_width, x2 + padding)
                pos_y = y1
            elif strategy == "left-outside":
                pos_x = max(0, x1 - label_width - padding)
                pos_y = y1
            else:
                pos_x = x1
                pos_y = max(0, y1 - label_height - padding)

            # Clamp to image boundaries
            pos_x = max(0, min(img_width - label_width, pos_x))
            pos_y = max(0, min(img_height - label_height, pos_y))

            # Calculate label bbox at this position
            label_bbox = (pos_x, pos_y, pos_x + label_width, pos_y + label_height)

            # Check if this position overlaps with any existing labels
            overlaps = any(_labels_overlap(label_bbox, existing) for existing in label_positions_bboxes)

            if not overlaps:
                # Found non-overlapping position
                label_positions_xy.append((pos_x, pos_y))
                label_positions_bboxes.append(label_bbox)
                found_position = True
                break

        # If all strategies failed, use offset fallback
        if not found_position:
            offset = (i % 10) * 8  # Stagger overlapping labels
            pos_x = x1 + offset
            pos_y = max(0, y1 - label_height - padding) + offset
            pos_x = max(0, min(img_width - label_width, pos_x))
            pos_y = max(0, min(img_height - label_height, pos_y))

            label_bbox = (pos_x, pos_y, pos_x + label_width, pos_y + label_height)
            label_positions_xy.append((pos_x, pos_y))
            label_positions_bboxes.append(label_bbox)

    return label_positions_xy


def draw_bounding_boxes(
    image: Image.Image,
    bboxes: List[List[float]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    thickness: int = 3,
) -> Image.Image:
    """Draw bounding boxes with labels on an image.

    Automatically positions labels to avoid overlaps when bboxes are close together.

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

    # Calculate non-overlapping label positions for all bboxes
    label_positions = _find_non_overlapping_positions(bboxes, labels, image.size, font)

    # Draw bounding boxes and labels
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, bbox)
        color = colors[i]
        label = labels[i]

        # Draw rectangle with thick outline
        for offset in range(thickness):
            # Calculate inner coordinates with offset
            inner_x1 = x1 + offset
            inner_y1 = y1 + offset
            inner_x2 = x2 - offset
            inner_y2 = y2 - offset

            # Skip if box would become inverted (too small for more offset layers)
            if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
                break

            draw.rectangle(
                [inner_x1, inner_y1, inner_x2, inner_y2],
                outline=color,
                width=1,
            )

        # Get pre-calculated label position
        label_x, label_y = label_positions[i]

        # Draw label background
        label_bbox = draw.textbbox((label_x, label_y), label, font=font)
        draw.rectangle(label_bbox, fill=color)

        # Draw label text
        draw.text((label_x, label_y), label, fill="white", font=font)

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
