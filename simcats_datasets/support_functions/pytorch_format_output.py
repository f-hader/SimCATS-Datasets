"""Functions for formatting the output of the **Pytorch Dataset class**.

Every function must accept a measurement (as array), a ground truth (e.g. TCT mask as array) and the image id as input.
Output type depends on the ground truth type and the required pytorch datatype (tensor as long, float, ...). Ground
truth could for example be a pixel mask or defined start end points of lines.
**Please look at format_dict_csd_float_ground_truth_long for a reference.**

@author: f.hader
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

def format_dict_csd_float_ground_truth_long(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float and long, respectively. (default of Pytorch Dataset class.)

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the measurement. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float and long, respectively.
    """
    assert (measurement.size == ground_truth.size), \
        f"Image and mask should be the same size, but are {measurement.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(measurement.copy(), dtype=torch.float).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.long, ).contiguous(), }


def format_dict_csd_float16_ground_truth_long(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float16 and long, respectively.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the measurement. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float16 and long, respectively.
    """
    assert (measurement.size == ground_truth.size), \
        f"Image and mask should be the same size, but are {measurement.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(measurement.copy(), dtype=torch.float16).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.long, ).contiguous(), }


def format_dict_csd_float_ground_truth_float(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float and float, respectively.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the measurement. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float and float, respectively.
    """
    assert (measurement.size == ground_truth.size), \
        f"Image and mask should be the same size, but are {measurement.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(measurement.copy(), dtype=torch.float).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.float).contiguous(), }


def format_mmsegmentation(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be conform to the MMSegmentation CustomDataset of version 0.6.0, see https://github.com/open-mmlab/mmsegmentation/blob/v0.6.0/mmseg/datasets/custom.py.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the measurement.

    Returns:
        Dict with data conform to the MMSegmentation CustomDataset of version 0.6.0, see https://github.com/open-mmlab/mmsegmentation/blob/v0.6.0/mmseg/datasets/custom.py.
    """
    assert (measurement.size == ground_truth.size), \
        f"Image and mask should be the same size, but are {measurement.size=} and {ground_truth.size=}"
    return {"img": torch.as_tensor(measurement.copy()).float().contiguous(),
        "gt_semantic_seg": torch.as_tensor(ground_truth.copy()).float().contiguous(),
        "img_metas": {"filename": f"{idx}.jpg", "ori_filename": f"{idx}_ori.jpg", "ori_shape": measurement.shape[::-1],
            # we want (100, 100, 1) not (1, 100, 100)
            "img_shape": measurement.shape[::-1], "pad_shape": measurement.shape[::-1],  # image shape after padding
            "scale_factor": 1.0, "img_norm_cfg": {"mean": np.mean(measurement, axis=(-2, -1)),  # mean for each channel
                "std": np.std(measurement, axis=(-2, -1)),  # std for each channel
                "to_rgb": False, }, "img_id": f"{idx}", }, }


def format_csd_only(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a measurement.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the measurement. Not used in this format.

    Returns:
        The measurement as tensor.
    """
    return torch.as_tensor(measurement.copy(), dtype=torch.float).contiguous()


def format_csd_float16_only(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a float16 (half precision) measurement.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the measurement. Not used in this format.

    Returns:
        The float 16 (half precision) measurement as tensor.
    """
    return torch.as_tensor(measurement.copy(), dtype=torch.float16).contiguous()


def format_csd_bfloat16_only(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a bfloat16 (half precision) measurement.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the measurement. Not used in this format.

    Returns:
        The brain float 16 (half precision) measurement as tensor.
    """
    return torch.as_tensor(measurement.copy(), dtype=torch.bfloat16).contiguous()


def format_csd_class_index(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> Tuple[
    torch.Tensor, torch.Tensor, int]:
    """Format the output of the Pytorch Dataset class to be the measurement, a class index (which is always 0 as we have no classes) and the index.

    This is needed to be conform to the datasets used in DeepSVDD, see https://github.com/lukasruff/Deep-SVDD-PyTorch.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the measurement.

    Returns:
        A tuple of measurement, class index, and the index.
    """
    return torch.as_tensor(measurement.copy(), dtype=torch.float).unsqueeze(0).contiguous(), torch.tensor(0), idx


def format_tuple_csd_float_ground_truth_float(measurement: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a tuple of the measurement and the ground_truth.

    Args:
        measurement: The measurement array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the measurement. Not used in this format.

    Returns:
        Tuple with measurement and ground_truth of dtype float and float, respectively.
    """
    assert (measurement.size == ground_truth.size), \
        f"Image and mask should be the same size, but are {measurement.size=} and {ground_truth.size=}"
    return (torch.as_tensor(measurement.copy(), dtype=torch.float).contiguous(),
            torch.as_tensor(ground_truth.copy(), dtype=torch.float).contiguous(),)
