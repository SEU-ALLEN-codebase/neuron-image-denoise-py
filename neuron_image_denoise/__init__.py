from .filter import adaptive_denoise
import numpy as np


def adaptive_denoise_16to8(img: np.ndarray, lower: int | float = 0, upper: int | float = 255, **kwargs):
    """
    The adaptive denoising is designed for 16bit raw image with full details.
    This wrapper appends a bit conversion afterward, by clipping and linear scaling.

    keyword arguments are passed to `adaptive_denoise`

    :param img: 3D neuron fluorescent image array, 16bit.
    :param lower: the lower threshold or quantile
    :param upper: the lower threshold or quantile
    :return: denoised 3D image array, 8bit.
    """
    assert img.dtype == np.uint16
    assert img.ndim == 3
    img = adaptive_denoise(img, **kwargs)
    if type(lower) is float:
        lower = np.quantile(img, lower)
    if type(upper) is float:
        upper = np.quantile(img, upper)
    return ((img.clip(lower, upper) - lower) / (upper - lower)).astype(np.uint8)
