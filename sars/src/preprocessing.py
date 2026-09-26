import numpy as np


def to_db(power, floor=1e-10):
    """
    Convert SAR power/backscatter to decibels.

    dB = 10 * log10(power)
    """
    power = np.asarray(power, dtype=np.float32)
    power = np.maximum(power, floor)

    return 10.0 * np.log10(power)


def normalize(array):
    """
    Normalize an array to approximately [0, 1].
    """
    array = np.asarray(array, dtype=np.float32)

    minimum = np.nanmin(array)
    maximum = np.nanmax(array)

    if maximum == minimum:
        return np.zeros_like(array)

    return (array - minimum) / (maximum - minimum)
