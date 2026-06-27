"""
Shared image processing utilities for SILO.

Common operations extracted from the main comparison pipeline to reduce
code duplication and provide a consistent interface for image manipulation.
"""

from typing import Tuple

import numpy as np
import cv2


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale if it has color channels.

    Args:
        img: Input image (RGB or grayscale).

    Returns:
        Grayscale image as a 2D array.
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def resize_to_max(img: np.ndarray, max_size: int) -> np.ndarray:
    """Resize image so its largest dimension does not exceed max_size.

    Preserves aspect ratio. Returns the image unchanged if already within
    the size limit.

    Args:
        img: Input image.
        max_size: Maximum allowed dimension (width or height).

    Returns:
        Resized image (or original if no resize needed).
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resize_to_match(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize image to match target (height, width) dimensions.

    Args:
        img: Input image to resize.
        target_shape: Desired (height, width).

    Returns:
        Resized image matching the target dimensions.
    """
    target_h, target_w = target_shape
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def compute_pixel_diff(img1: np.ndarray, img2: np.ndarray,
                       threshold: int = 15) -> Tuple[float, np.ndarray]:
    """Compute pixel difference percentage and a binary change mask.

    Takes the absolute per-pixel difference, reduces multi-channel images
    to the max channel difference, then thresholds to produce a binary mask.

    Args:
        img1: First image (same dimensions as img2).
        img2: Second image (same dimensions as img1).
        threshold: Pixel intensity difference threshold for the change mask.

    Returns:
        Tuple of (percentage of pixels that changed, binary change mask).
    """
    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    if len(diff.shape) == 3:
        diff_gray = np.max(diff, axis=2)
    else:
        diff_gray = diff

    change_mask = (diff_gray >= threshold).astype(np.uint8)
    pixel_diff_pct = (np.sum(change_mask) / change_mask.size) * 100

    return pixel_diff_pct, change_mask


def compute_size_ratio(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> float:
    """Compute the area ratio between two image shapes.

    Returns a value in (0, 1] where 1 means identical area and smaller
    values indicate greater size disparity.

    Args:
        shape1: Shape tuple of first image (at least height, width).
        shape2: Shape tuple of second image (at least height, width).

    Returns:
        Ratio of the smaller area to the larger area.
    """
    area1 = shape1[0] * shape1[1]
    area2 = shape2[0] * shape2[1]
    return min(area1, area2) / max(area1, area2)
