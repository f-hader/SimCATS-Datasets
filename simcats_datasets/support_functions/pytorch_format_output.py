"""Functions for formatting the output of the **Pytorch Dataset class**.

Every function must accept a CSD (as array), a ground truth (e.g. TCT mask as array) and the image id as input.
Output type depends on the ground truth type and the required pytorch datatype (tensor as long, float, ...). Ground
truth could for example be a pixel mask or defined start end points of lines.
**Please look at format_dict_csd_float_ground_truth_long for a reference.**

@author: f.hader
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def format_dict_csd_float_ground_truth_long(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float and long, respectively. (default of Pytorch Dataset class.)

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the csd. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float and long, respectively.
    """
    assert (
            csd.size == ground_truth.size), f"Image and mask should be the same size, but are {csd.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(csd.copy(), dtype=torch.float).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.long, ).contiguous(), }


def format_dict_csd_float16_ground_truth_long(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float16 and long, respectively.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the csd. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float16 and long, respectively.
    """
    assert (
            csd.size == ground_truth.size), f"Image and mask should be the same size, but are {csd.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(csd.copy(), dtype=torch.float16).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.long, ).contiguous(), }


def format_dict_csd_float_ground_truth_float(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a dict with entries 'csd' and 'ground_truth' of dtype float and float, respectively.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the csd. Not used in this format.

    Returns:
        Dict with 'csd' and 'ground_truth' of dtype float and float, respectively.
    """
    assert (
            csd.size == ground_truth.size), f"Image and mask should be the same size, but are {csd.size=} and {ground_truth.size=}"
    return {"csd": torch.as_tensor(csd.copy(), dtype=torch.float).contiguous(),
        "ground_truth": torch.as_tensor(ground_truth.copy(), dtype=torch.float).contiguous(), }


def format_mmsegmentation(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be conform to the MMSegmentation CustomDataset of version 0.6.0, see https://github.com/open-mmlab/mmsegmentation/blob/v0.6.0/mmseg/datasets/custom.py.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the csd.

    Returns:
        Dict with data conform to the MMSegmentation CustomDataset of version 0.6.0, see https://github.com/open-mmlab/mmsegmentation/blob/v0.6.0/mmseg/datasets/custom.py.
    """
    assert (
            csd.size == ground_truth.size), f"Image and mask should be the same size, but are {csd.size=} and {ground_truth.size=}"
    return {"img": torch.as_tensor(csd.copy()).float().contiguous(),
        "gt_semantic_seg": torch.as_tensor(ground_truth.copy()).float().contiguous(),
        "img_metas": {"filename": f"{idx}.jpg", "ori_filename": f"{idx}_ori.jpg", "ori_shape": csd.shape[::-1],
            # we want (100, 100, 1) not (1, 100, 100)
            "img_shape": csd.shape[::-1], "pad_shape": csd.shape[::-1],  # image shape after padding
            "scale_factor": 1.0, "img_norm_cfg": {"mean": np.mean(csd, axis=(-2, -1)),  # mean for each channel
                "std": np.std(csd, axis=(-2, -1)),  # std for each channel
                "to_rgb": False, }, "img_id": f"{idx}", }, }


def format_csd_only(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a CSD.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the csd. Not used in this format.

    Returns:
        The CSD as tensor.
    """
    return torch.as_tensor(csd.copy(), dtype=torch.float).contiguous()


def format_csd_float16_only(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a float16 (half precision) CSD.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the csd. Not used in this format.

    Returns:
        The float 16 (half precision) CSD as tensor.
    """
    return torch.as_tensor(csd.copy(), dtype=torch.float16).contiguous()


def format_csd_bfloat16_only(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> torch.Tensor:
    """Format the output of the Pytorch Dataset class to be just a bfloat16 (half precision) CSD.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the csd. Not used in this format.

    Returns:
        The brain float 16 (half precision) CSD as tensor.
    """
    return torch.as_tensor(csd.copy(), dtype=torch.bfloat16).contiguous()


def format_csd_class_index(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> Tuple[
    torch.Tensor, torch.Tensor, int]:
    """Format the output of the Pytorch Dataset class to be the CSD, a class index (which is always 0 as we have no classes) and the index.

    This is needed to be conform to the datasets used in DeepSVDD, see https://github.com/lukasruff/Deep-SVDD-PyTorch.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask. Not used in this format.
        idx: Index of the csd.

    Returns:
        A tuple of CSD, class index, and the index.
    """
    return torch.as_tensor(csd.copy(), dtype=torch.float).unsqueeze(0).contiguous(), torch.tensor(0), idx


def format_tuple_csd_float_ground_truth_float(csd: np.ndarray, ground_truth: np.ndarray, idx: int, ) -> dict[
    str, torch.Tensor]:
    """Format the output of the Pytorch Dataset class to be a tuple of the csd and the ground_truth.

    Args:
        csd: The CSD array.
        ground_truth: Ground truth as pixel mask.
        idx: index of the csd. Not used in this format.

    Returns:
        Tuple with csd and ground_truth of dtype float and float, respectively.
    """
    assert (
            csd.size == ground_truth.size), f"Image and mask should be the same size, but are {csd.size=} and {ground_truth.size=}"
    return (torch.as_tensor(csd.copy(), dtype=torch.float).contiguous(),
            torch.as_tensor(ground_truth.copy(), dtype=torch.float).contiguous(),)
