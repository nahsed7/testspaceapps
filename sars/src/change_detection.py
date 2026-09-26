import numpy as np


def difference(image1, image2):
    """
    Pixel-wise difference between two SAR images.
    """
    image1 = np.asarray(image1, dtype=np.float32)
    image2 = np.asarray(image2, dtype=np.float32)

    if image1.shape != image2.shape:
        raise ValueError("SAR images must have the same shape.")

    return image2 - image1


def absolute_difference(image1, image2):
    """
    Magnitude of pixel-wise change.
    """
    return np.abs(difference(image1, image2))


def threshold_change(change_map, threshold):
    """
    Create a binary change mask.
    """
    return np.abs(change_map) >= threshold
