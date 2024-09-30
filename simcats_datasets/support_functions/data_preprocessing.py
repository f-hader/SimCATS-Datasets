"""Data preprocessors to be used with the **Pytorch Dataset class**.

Every preprocessor must accept either a single array or a list of arrays as input. Output type should always be the same
as the input type. Please try to use -=, +=, *=, and /=, as these are way faster than data = data + ... etc.. Avoid
using map(function, data), as this will return a copy and copying will slow down your code.
**Please look at example_preprocessor for a reference.**
"""

from typing import List, Union, Tuple

import numpy as np
import cv2
import skimage.restoration
import bm3d
from scipy.signal import resample, decimate


def example_preprocessor(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Example (reference) for preprocessor implementations.

    Args:
        data: Numpy array to be preprocessed (or a list of such).

    Returns:
        Preprocessed numpy array (or a list of such).
    """
    # handle list here, for example with list comprehension
    if isinstance(data, list):
        data = [_data for _data in data]
    else:
        data = data
    return data


def cast_to_float32(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Cast the data to float32. Especially useful to reduce memory usage for preloaded datasets.

    Args:
        data: Numpy array to be cast to float32 (or a list of such).

    Returns:
        Float32 numpy array (or a list of such).
    """
    # handle list here, for example with list comprehension
    if isinstance(data, list):
        data = [_data.astype(np.float32) for _data in data]
    else:
        data = data.astype(np.float32)
    return data


def cast_to_float16(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Cast the data to float16. Especially useful to reduce memory usage for preloaded datasets.

    Args:
        data: Numpy array to be cast to float16 (or a list of such).

    Returns:
        Float16 numpy array (or a list of such).
    """
    # handle list here, for example with list comprehension
    if isinstance(data, list):
        data = [_data.astype(np.float16) for _data in data]
    else:
        data = data.astype(np.float16)
    return data


def standardization(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Standardization of the data (mean=0, std=1).

    If a list of data is passed, each data is standardized individually (no global standardization).

    Args:
        data: Numpy array to be standardized (or a list of such).

    Returns:
        Standardized numpy array (or a list of such).
    """
    if isinstance(data, list):
        for _data in data:
            _data -= np.mean(_data)
            _data /= np.std(_data)
    else:
        data -= np.mean(data)
        data /= np.std(data)
    return data


def min_max_0_1(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Min max scaling of the data to [0, 1].

    If a list of data is passed, each data is scaled individually (no global scaling).

    Args:
        data: Numpy array to be scaled (or a list of such).

    Returns:
        Rescaled numpy array (or a list of such).
    """
    if isinstance(data, list):
        for _data in data:
            _data -= np.min(_data)
            _data /= np.max(_data)
    else:
        data -= np.min(data)
        data /= np.max(data)
    return data


def min_max_minus_one_one(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Min max scaling of the data to [-1, 1].

    If a list of data is passed, each data is scaled individually (no global scaling).

    Args:
        data: Numpy array to be scaled (or a list of such).

    Returns:
        Rescaled numpy array (or a list of such).
    """
    data = min_max_0_1(data)
    if isinstance(data, list):
        for _data in data:
            _data -= 0.5
            _data *= 2
    else:
        data -= 0.5
        data *= 2
    return data


def add_newaxis(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Adds a new axis to the data (basically the missing color channel).

    Args:
        data: Numpy array to which the axis will be added (or a list of such).

    Returns:
        Numpy array with additional axis (or a list of such).
    """
    if isinstance(data, list):
        return [_data[np.newaxis, ...] for _data in data]
    return data[np.newaxis, ...]


def only_two_classes(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Sets all mask labels that are larger than or equal 1 to 1 and all other pixels to zero.

    Args:
        data: Numpy array to be processed (or a list of such).

    Returns:
        Numpy array with only two classes (or a list of such).
    """
    if isinstance(data, list):
        for _data in data:
            _data[_data >= 1] = 1
            _data[_data < 1] = 0
    else:
        data[data >= 1] = 1
        data[data < 1] = 0
    return data


def shrink_to_shape(data: Union[np.ndarray, List[np.ndarray]], shape: Tuple[int, int]) -> Union[
    np.ndarray, List[np.ndarray]]:
    """Cut off required number of rows/columns of pixels at each edge of the image to get the desired shape.

    **Warning**: This preprocessor can't be used by supplying a string with the name to the class SimcatsDataset from
    the simcats_datasets.pytorch module, as this requires that preprocessors need no additional parameters but only the
    data. If a list of data is passed, it is expected, that all images in the list have the same shape!

    Args:
        data: Numpy array to be preprocessed (or a list of such).
        shape: The shape to which the data will be reshaped.

    Returns:
        Shrinked numpy array (or a list of such).
    """
    if isinstance(data, list) and data[0].shape != shape:
        axis0_start = (data[0].shape[0] - shape[0]) // 2
        axis0_stop = -data[0].shape[0] + shape[0] + axis0_start
        axis1_start = (data[0].shape[1] - shape[1]) // 2
        axis1_stop = -data[0].shape[1] + shape[1] + axis1_start
        data = [_data[axis0_start:axis0_stop, axis1_start:axis1_stop] for _data in data]
    elif data.shape != shape:
        axis0_start = (data.shape[0] - shape[0]) // 2
        axis0_stop = -data.shape[0] + shape[0] + axis0_start
        axis1_start = (data.shape[1] - shape[1]) // 2
        axis1_stop = -data.shape[1] + shape[1] + axis1_start
        data = data[axis0_start:axis0_stop, axis1_start:axis1_stop]
    return data


def shrink_to_shape_96x96(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Cut off required number of rows/columns of pixels at each edge of the image to get shape 96x96.

    **Warning**: If a list of data is passed, it is expected, that all images in the list have the same shape!

    Args:
        data: Numpy array to be preprocessed (or a list of such).

    Returns:
        Shrinked numpy array (or a list of such).
    """
    return shrink_to_shape(data=data, shape=(96, 96))


def resample_image(data: Union[np.ndarray, List[np.ndarray]], target_size: Tuple[int, int]) -> Union[
    np.ndarray, List[np.ndarray]]:
    """Resample an image to target size using scipy.signal.resample.

    **Warning**: This preprocessor can't be used by supplying a string with the name to the class SimcatsDataset from
    the simcats_datasets.pytorch module, as it requires that preprocessors need no additional parameters but only the
    data.

    Args:
        data: The image to resample.
        target_size: The target size to resample to.

    Returns:
        The resampled image or a list of such.
    """
    if isinstance(data, list):
        data = [resample_image(temp_data) for temp_data in data]
    else:
        if data.shape[0] > target_size[0]:
            data = resample(data, target_size[0], axis=0)
        if data.shape[1] > target_size[1]:
            data = resample(data, target_size[1], axis=1)
    return data


def decimate_image(data: Union[np.ndarray, List[np.ndarray]], target_size: Tuple[int, int]) -> Union[
    np.ndarray, List[np.ndarray]]:
    """Decimate an image to target size using scipy.signal.decimate.

    **Warning**: This preprocessor can't be used by supplying a string with the name to the class SimcatsDataset from
    the simcats_datasets.pytorch module, as it requires that preprocessors need no additional parameters but only the
    data.

    Args:
        data: The image to decimate.
        target_size: The target size to decimate to.

    Returns:
        The decimated image or a list of such.
    """
    if isinstance(data, list):
        data = [decimate_image(temp_data) for temp_data in data]
    else:
        q = [data.shape[0] / target_size[0], data.shape[1] / target_size[1]]
        while q[0] > 1 or q[1] > 1:
            if q[0] > 1:
                data = decimate(data.T, min(13, int(np.ceil(q[0]))), axis=1, ftype="iir").T
            if q[1] > 1:
                data = decimate(data.T, min(13, int(np.ceil(q[1]))), axis=0, ftype="iir").T
            q = [data.shape[0] / target_size[0], data.shape[1] / target_size[1]]
    return data


def standardize_to_dataset(data: Union[np.ndarray, List[np.ndarray]], mean: float, std: float) -> Union[
    np.ndarray, List[np.ndarray]]:
    """Standardization of the data not per image but for a whole dataset.

    **Warning**: This preprocessor can't be used by supplying a string with the name to the class SimcatsDataset from
    the simcats_datasets.pytorch module, as it requires that preprocessors need no additional parameters but only the
    data.

    Args:
        data (Union[np.ndarray, List[np.ndarray]]):  Numpy array to be standardized (or a list of such).
        mean (float): The mean to subtract.
        std (float): The standard deviation to divide by.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Standardized numpy array (or a list of such).
    """
    if isinstance(data, list):
        for _data in data:
            _data -= mean
            _data /= std
    else:
        data -= mean
        data /= std

    return data


def _bm3d_smoothing_single_img(img: np.ndarray) -> np.ndarray:
    """BM3D smoothing helper function, which performs the actual BM3D smoothing in the bm3d_smoothing preprocessor.

    Args:
        img: Numpy array to be smoothed.

    Returns:
        Smoothed image.
    """
    sigma = 0.4 * skimage.restoration.estimate_sigma(img, average_sigmas=True)
    img = bm3d.bm3d(img, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return img


def bm3d_smoothing(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Smoothing of the data using the BM3D algorithm.
    
    Args:
        data: Numpy array to be smoothed (or a list of such)
    
    Returns:
        BM3D-smoothed numpy array (or a list of such)
    """
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = _bm3d_smoothing_single_img(data[i])
    else:
        data = _bm3d_smoothing_single_img(data)
    return data


def _vertical_median_smoothing_single_img(img: np.ndarray) -> np.ndarray:
    """Vertical median smoothing helper function, which performs the actual smoothing in the vertical_median_smoothing preprocessor.

    Args:
        img: Numpy array to be smoothed.

    Returns:
        Smoothed image.
    """
    for i in range(img.shape[1]):
        img[:, i] = cv2.medianBlur(img[:, i], 3).flatten()
    return img


def vertical_median_smoothing(data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
    """Median-smoothing of the data, for each vertical column independently.

    Args:
        data: Numpy array to be smoothed (or a list of such).

    Returns:
        Smoothed numpy array (or a list of such).
    """
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i].astype(np.float32)
            data[i] = _vertical_median_smoothing_single_img(data[i])
    else:
        data = data.astype(np.float32)
        data = _vertical_median_smoothing_single_img(data)
    return data
