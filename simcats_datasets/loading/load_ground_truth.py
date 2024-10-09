"""Functions for providing ground truth data to be used with the **Pytorch Dataset class**.

For examples of the different ground truth types, please have a look at the notebook Examples_Pytorch_SimcatsDataset.

Every function must accept a h5 File or path for a simcats_dataset as input, provide an option to use only specific_ids
and allow disabling the progress_bar.
Output type depends on the ground truth type. Could for example be a pixel mask or defined start end points of lines.
**Please look at load_zeros_masks for a reference.**

@author: f.hader
"""

from typing import Union, List

import bezier
import h5py
import numpy as np

# imports required for eval, to create IdealCSDGeometric objects from metadata strings
import re

import sympy
from simcats.ideal_csd import IdealCSDGeometric
from simcats.ideal_csd.geometric import calculate_all_bezier_anchors, tct_bezier, initialize_tct_functions
from numpy import array
from simcats.distortions import OccupationDotJumps
from tqdm import tqdm

from simcats_datasets.loading import load_dataset
from simcats.support_functions import rotate_points


# Lists defining which ground truth type is supported for CSD and sensor scan datasets, respectively
_csd_ground_truths = ["load_zeros_masks", "load_tct_masks", "load_tct_by_dot_masks", "load_idt_masks", "load_ct_masks",
                      "load_ct_by_dot_masks", "load_tc_region_masks", "load_tc_region_minus_tct_masks",
                      "load_c_region_masks"]
_sensor_scan_ground_truths = ["load_zeros_masks", "load_tct_masks"]


def load_zeros_masks(file: Union[str, h5py.File],
                     specific_ids: Union[range, List[int], np.ndarray, None] = None,
                     progress_bar: bool = True) -> List[np.ndarray]:
    """Load no/empty ground truth data (arrays with only zeros).
    Used for loading sets without ground truth. This is helpful to e.g. load experimental datasets without labels with
    the pytorch SimcatsDataset class to analyze train results with the same Interface as for simulated data.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        List of arrays containing only zeros as ground truth data
    """
    return [np.zeros_like(csd, dtype=np.uint8) for csd in
            load_dataset(file=file, load_csds=True, specific_ids=specific_ids, progress_bar=progress_bar).csds]


def load_tct_masks(file: Union[str, h5py.File],
                   specific_ids: Union[range, List[int], np.ndarray, None] = None,
                   progress_bar: bool = True) -> List[np.ndarray]:
    """Load Total Charge Transition (TCT) masks as ground truth data.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Total Charge Transition (TCT) masks
    """
    return load_dataset(file=file, load_csds=False, load_tct_masks=True, specific_ids=specific_ids,
                        progress_bar=progress_bar).tct_masks


def load_tct_by_dot_masks(file: Union[str, h5py.File],
                          specific_ids: Union[range, List[int], np.ndarray, None] = None,
                          progress_bar: bool = True,
                          lut_entries: int = 1000) -> List[np.ndarray]:
    """Load Total Charge Transition (TCT) masks with transitions labeled by affected dot as ground truth data.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.
        lut_entries: Number of lookup-table entries to use for tct_bezier. Default is 1000.

    Returns:
        Total Charge Transition (TCT) masks
    """
    tct_masks = load_dataset(file=file, load_csds=False, load_tct_masks=True, specific_ids=specific_ids,
                             progress_bar=progress_bar).tct_masks
    metadata = load_dataset(file=file, load_csds=False, load_metadata=True, specific_ids=specific_ids,
                            progress_bar=progress_bar).metadata

    for mask_id, meta in tqdm(enumerate(metadata), desc="calculating transitions", total=len(metadata),
                              disable=not progress_bar):
        try:
            csd_geometric = eval(meta["ideal_csd_config"])
        except:
            # This is required to support metadata from older simcats versions, where the class had a different name
            csd_geometric = eval(meta["ideal_csd_config"].replace("IdealCSDGeometrical", "IdealCSDGeometric"))

        tct_ids = np.unique(tct_masks[mask_id][np.nonzero(tct_masks[mask_id])])
        # skip images with no TCTs
        if tct_ids.size == 0:
            continue
        # setup tct functions
        tct_funcs = initialize_tct_functions(tct_params=csd_geometric.tct_params[np.min(tct_ids) - 1:np.max(tct_ids)],
                                             max_peaks=np.min(tct_ids))
        # calculate all bezier anchors
        bezier_coords = calculate_all_bezier_anchors(
            tct_params=csd_geometric.tct_params[np.min(tct_ids) - 1:np.max(tct_ids)], max_peaks=np.min(tct_ids),
            rotation=csd_geometric.rotation)

        # get parameters of pixel vs voltage space for discretization
        x_res = tct_masks[mask_id].shape[1]
        y_res = tct_masks[mask_id].shape[0]
        x_lims = meta["sweep_range_g1"]
        y_lims = meta["sweep_range_g2"]
        # stepsize x/y
        x_step = (x_lims[-1] - x_lims[0]) / (x_res - 1)
        y_step = (y_lims[-1] - y_lims[0]) / (y_res - 1)

        # get corner points to know max value range for generating points
        corner_points = np.array(
            [[x_lims[0], y_lims[0]], [x_lims[0], y_lims[1]], [x_lims[1], y_lims[0]], [x_lims[1], y_lims[1]]])
        x_c_rot = rotate_points(points=corner_points, angle=-csd_geometric.rotation)[:, 0]

        # replace tct_mask by a new empty array
        tct_masks[mask_id] = np.zeros_like(tct_masks[mask_id], dtype=np.uint8)
        for tct_id in tct_ids:
            for transition in range(tct_id * 2):
                # get start x position of current transition
                if transition == 0:
                    x_start = np.min(x_c_rot) - x_step
                else:
                    # rotate bezier coords
                    bezier_coords_rot = rotate_points(points=bezier_coords[tct_id][transition - 1, 1],
                                                      angle=-csd_geometric.rotation)
                    x_start = bezier_coords_rot[0]
                # get stop x position of current transition
                if transition == tct_id * 2 - 1:
                    x_stop = np.max(x_c_rot) + x_step
                else:
                    # rotate bezier coords
                    bezier_coords_rot = rotate_points(points=bezier_coords[tct_id][transition, 1],
                                                      angle=-csd_geometric.rotation)
                    x_stop = bezier_coords_rot[0]

                # generate enough x-values to cover the complete range of the CSD with a
                # higher resolution than required to have a precise result after discretization
                tct_points = np.empty(((x_res + y_res) * 4, 2))
                tct_points[:, 0] = np.linspace(x_start, x_stop, (x_res + y_res) * 4)

                # Insert the transition line into the CSD
                # The required TCT points are sampled and discretized.
                # generate the y-values for all generated x-values
                tct_points[:, 1] = tct_funcs[tct_id](x_eval=tct_points[:, 0], lut_entries=lut_entries)

                # rotate the TCT into the original orientation
                wf_points_rot = rotate_points(points=tct_points, angle=csd_geometric.rotation)

                # select only TCT pixels that are in the csd-limits
                valid_ids = np.where((wf_points_rot[:, 0] > (x_lims[0] - 0.5 * x_step)) & (
                        wf_points_rot[:, 0] < (x_lims[1] + 0.5 * x_step)) & (
                                             wf_points_rot[:, 1] > (y_lims[0] - 0.5 * y_step)) & (
                                             wf_points_rot[:, 1] < (y_lims[1] + 0.5 * y_step)))
                # x_h_rot = x_h_rot[valid_ids]
                # y_h_rot = y_h_rot[valid_ids]
                wf_points_rot = wf_points_rot[valid_ids[0], :]

                # insert TCT pixels into the csd
                # calculation of the ids for the values:
                # x = min(csd_x) + id * x_step
                # add half step size, so that the pixel id of the nearest pixel is obtained after the division
                # (round up if next higher value in range of 0.5 * step_size)
                x_id = np.floor_divide(wf_points_rot[:, 0] + 0.5 * x_step - x_lims[0], x_step).astype(int)
                y_id = np.floor_divide(wf_points_rot[:, 1] + 0.5 * y_step - y_lims[0], y_step).astype(int)
                tct_masks[mask_id][y_id, x_id] = ((transition + 1) % 2) + 1

        # apply dot jumps if any were active
        if "OccupationDotJumps_axis0" in meta or "OccupationDotJumps_axis1" in meta:
            occ_jumps_objects_string = [s.split(", rng")[0] + ")" for s in
                                        re.findall(r"OccupationDotJumps[^\]]*", meta["occupation_distortions"]) if
                                        "[activated" in s]
            occ_jumps_objects = [eval(s) for s in occ_jumps_objects_string]
            for obj in occ_jumps_objects:
                if f"OccupationDotJumps_axis{obj.axis}" in meta:
                    obj._OccupationDotJumps__activated = True
                    obj._OccupationDotJumps__previous_noise = meta[f"OccupationDotJumps_axis{obj.axis}"]
                    _, tct_masks[mask_id] = obj.noise_function(occupations=np.empty((0, 0, 2)),
                                                               lead_transitions=tct_masks[mask_id],
                                                               volt_limits_g1=x_lims, volt_limits_g2=y_lims,
                                                               freeze=True)
    return tct_masks


def load_idt_masks(file: Union[str, h5py.File],
                   specific_ids: Union[range, List[int], np.ndarray, None] = None,
                   progress_bar: bool = True) -> List[np.ndarray]:
    """Load Inter-Dot Transition (IDT) masks as ground truth data.
    In comparison to the Total Charge Transition (TCT) masks, only inter-dot transitions are included.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Inter-Dot Transition (IDT) masks
    """
    tct_masks = load_dataset(file=file, load_csds=False, load_tct_masks=True, specific_ids=specific_ids,
                             progress_bar=progress_bar).tct_masks
    metadata = load_dataset(file=file, load_csds=False, load_metadata=True, specific_ids=specific_ids,
                            progress_bar=progress_bar).metadata
    idt_masks = []
    for tct_mask, meta in tqdm(zip(tct_masks, metadata), desc="calculating idt", total=len(tct_masks),
                               disable=not progress_bar):
        idt_mask = np.zeros(tct_mask.shape, dtype=np.uint8)
        try:
            csd_geometric = eval(meta["ideal_csd_config"])
        except:
            # This is required to support metadata from older simcats versions, where the class had a different name
            csd_geometric = eval(meta["ideal_csd_config"].replace("IdealCSDGeometrical", "IdealCSDGeometric"))
        bezier_coords = calculate_all_bezier_anchors(tct_params=csd_geometric.tct_params[:int(np.max(tct_mask) + 1)],
                                                     rotation=csd_geometric.rotation)
        # get parameters of pixel vs voltage space for discretization
        x_res = idt_mask.shape[1]
        y_res = idt_mask.shape[0]
        x_lims = meta["sweep_range_g1"]
        y_lims = meta["sweep_range_g2"]
        # stepsize x/y
        x_step = (x_lims[-1] - x_lims[0]) / (x_res - 1)
        y_step = (y_lims[-1] - y_lims[0]) / (y_res - 1)
        try:
            # min tct minus 1, because we always need to start with the one below of the lowest in the image
            first_tct = int(np.max([np.min(tct_mask[np.nonzero(tct_mask)]) - 1, 1]))
        except ValueError:
            first_tct = 1
        # iterate over all tcts that are part of the current voltage range
        for i in range(first_tct, int(np.max(tct_mask) + 1)):
            # The number of inter-dot transitions connected to the next higher TCT for every TCT is given by the ID of the TCT
            for j in range(i):
                # get start and end vector for inter dot transitions
                inter_dot_transition = np.array([bezier_coords[i + 1][j * 2 + 1, 1, :], bezier_coords[i][j * 2, 1, :]])

                # sample points between the two outer bezier anchors of the current TCT to find the intersection with the interdot vector
                # (localized by the center bezier anchor). This is required because interdot transitions can be longer than
                # the distance between the central anchors (length depends on rounding, more rounding = longer)
                # rotate interdot transition into default representation
                inter_dot_transition_rot = rotate_points(points=inter_dot_transition, angle=-csd_geometric.rotation)

                # distance to lower TCT
                bezier_coords_rot = rotate_points(points=bezier_coords[i][j * 2], angle=-csd_geometric.rotation)
                x_eval = np.linspace(bezier_coords_rot[0, 0], bezier_coords_rot[2, 0], np.max(idt_mask.shape))
                # bezier nodes as fortran array
                nodes = np.asfortranarray(bezier_coords_rot.T)
                # initialize bezier curve
                bezier_curve = bezier.Curve.from_nodes(nodes)
                t = np.linspace(0, 1, np.max(idt_mask.shape) * 2)
                bezier_lut = bezier_curve.evaluate_multi(t)
                y_eval = [bezier_lut[1, np.argmin(np.abs(bezier_lut[0, :] - x))] for x in x_eval]
                # find the closest point: y_eval - (central_tct_anchor_y - (central_tct_anchor_x - x_eval) * (interdot_vec_y / interdot_vec_x))
                y_res = np.abs(y_eval - (bezier_coords_rot[1, 1] - (bezier_coords_rot[1, 0] - x_eval) * (
                        (inter_dot_transition_rot[0][1] - inter_dot_transition_rot[1][1]) / (
                        inter_dot_transition_rot[0][0] - inter_dot_transition_rot[1][0]))))
                intersection_pixel = np.argmin(y_res)
                # calculate the distance from this point to the central bezier anchor
                intersection_dist_to_bezier = np.linalg.norm(
                    [x_eval[intersection_pixel] - inter_dot_transition_rot[1][0],
                     y_eval[intersection_pixel] - inter_dot_transition_rot[1][1]])
                intersection_dist_to_lower_bezier_percentage = intersection_dist_to_bezier / np.linalg.norm(
                    (inter_dot_transition_rot[1] - inter_dot_transition_rot[0]))

                # distance to upper TCT
                bezier_coords_rot = rotate_points(points=bezier_coords[i + 1][j * 2 + 1], angle=-csd_geometric.rotation)
                x_eval = np.linspace(bezier_coords_rot[0, 0], bezier_coords_rot[2, 0], np.max(idt_mask.shape))
                # bezier nodes as fortran array
                nodes = np.asfortranarray(bezier_coords_rot.T)
                # initialize bezier curve
                bezier_curve = bezier.Curve.from_nodes(nodes)
                t = np.linspace(0, 1, np.max(idt_mask.shape) * 2)
                bezier_lut = bezier_curve.evaluate_multi(t)
                y_eval = [bezier_lut[1, np.argmin(np.abs(bezier_lut[0, :] - x))] for x in x_eval]
                # find the closest point: y_eval - (central_tct_anchor_y - (central_tct_anchor_x - x_eval) * (interdot_vec_y / interdot_vec_x))
                y_res = np.abs(y_eval - (bezier_coords_rot[1, 1] - (bezier_coords_rot[1, 0] - x_eval) * (
                        (inter_dot_transition_rot[0][1] - inter_dot_transition_rot[1][1]) / (
                        inter_dot_transition_rot[0][0] - inter_dot_transition_rot[1][0]))))
                intersection_pixel = np.argmin(y_res)
                # calculate the distance from this point to the central bezier anchor
                intersection_dist_to_bezier = np.linalg.norm(
                    [x_eval[intersection_pixel] - inter_dot_transition_rot[0][0],
                     y_eval[intersection_pixel] - inter_dot_transition_rot[0][1]])
                intersection_dist_to_upper_bezier_percentage = intersection_dist_to_bezier / np.linalg.norm(
                    (inter_dot_transition_rot[1] - inter_dot_transition_rot[0]))

                # sample some points along the transition
                inter_dot_points = inter_dot_transition[0] + \
                                   np.linspace(0 - intersection_dist_to_upper_bezier_percentage,
                                               1 + intersection_dist_to_lower_bezier_percentage,
                                               np.max(idt_mask.shape))[..., np.newaxis] * (
                                           inter_dot_transition[1] - inter_dot_transition[0])
                # select only pixels that are in the csd-limits
                valid_ids = np.where((inter_dot_points[:, 0] > (x_lims[0] - 0.5 * x_step)) & (
                        inter_dot_points[:, 0] < (x_lims[1] + 0.5 * x_step)) & (
                                             inter_dot_points[:, 1] > (y_lims[0] - 0.5 * y_step)) & (
                                             inter_dot_points[:, 1] < (y_lims[1] + 0.5 * y_step)))
                inter_dot_points = inter_dot_points[valid_ids[0], :]

                # insert pixels into the mask
                # calculation of the pixel ids for the values:
                # x = min(csd_x) + id * x_step
                # add half step size, so that the pixel id of the nearest pixel is obtained after the division
                # (round up if next higher value in range of 0.5 * step_size)
                x_id = np.floor_divide(inter_dot_points[:, 0] + 0.5 * x_step - x_lims[0], x_step).astype(int)
                y_id = np.floor_divide(inter_dot_points[:, 1] + 0.5 * y_step - y_lims[0], y_step).astype(int)
                idt_mask[y_id, x_id] = i
        # apply dot jumps if any were active
        if "OccupationDotJumps_axis0" in meta or "OccupationDotJumps_axis1" in meta:
            occ_jumps_objects_string = [s.split(", rng")[0] + ")" for s in
                                        re.findall(r"OccupationDotJumps[^\]]*", meta["occupation_distortions"]) if
                                        "[activated" in s]
            occ_jumps_objects = [eval(s) for s in occ_jumps_objects_string]
            for obj in occ_jumps_objects:
                if f"OccupationDotJumps_axis{obj.axis}" in meta:
                    obj._OccupationDotJumps__activated = True
                    obj._OccupationDotJumps__previous_noise = meta[f"OccupationDotJumps_axis{obj.axis}"]
                    _, idt_mask = obj.noise_function(occupations=np.empty((0, 0, 2)), lead_transitions=idt_mask,
                                                     volt_limits_g1=x_lims, volt_limits_g2=y_lims, freeze=True)
        idt_masks.append(idt_mask)
    return idt_masks


def load_ct_masks(file: Union[str, h5py.File],
                  specific_ids: Union[range, List[int], np.ndarray, None] = None,
                  progress_bar: bool = True) -> List[np.ndarray]:
    """Load Charge Transition (CT) masks as ground truth data.
    In comparison to the Total Charge Transition (TCT) masks, the inter-dot transitions are included.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Charge Transition (CT) masks
    """
    ct_masks = load_dataset(file=file, load_csds=False, load_tct_masks=True, specific_ids=specific_ids,
                            progress_bar=progress_bar).tct_masks
    idt_masks = load_idt_masks(file=file, specific_ids=specific_ids, progress_bar=progress_bar)
    for ct_mask, idt_mask in zip(ct_masks, idt_masks):
        ct_mask[(idt_mask > 0) & (ct_mask == 0)] = idt_mask[(idt_mask > 0) & (ct_mask == 0)]
    return ct_masks


def load_ct_by_dot_masks(file: Union[str, h5py.File],
                         specific_ids: Union[range, List[int], np.ndarray, None] = None,
                         progress_bar: bool = True,
                         lut_entries: int = 1000,
                         try_directly_loading_from_file: bool = True) -> List[np.ndarray]:
    """Load Charge Transition (CT) masks with transitions labeled by affected dot as ground truth data.
    In comparison to the Total Charge Transition (TCT) masks, the inter-dot transitions are included.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.
        lut_entries: Number of lookup-table entries to use for tct_bezier. Default is 1000.
        try_directly_loading_from_file: Specifies if the loader should try to find the masks in the h5 file before
            falling back to calculating them (not all datasets include these masks). Default is True.

    Returns:
        Charge Transition (CT) masks
    """
    ct_masks = None
    if try_directly_loading_from_file:
        try:
            ct_masks = load_dataset(file=file, load_csds=False, load_ct_by_dot_masks=True, specific_ids=specific_ids,
                                    progress_bar=progress_bar).ct_by_dot_masks
        except KeyError:
            pass
    # if the data could not be loaded from the file, or loading from the file was disabled, calculate it manually
    if ct_masks is None:
        ct_masks = load_tct_by_dot_masks(file=file, specific_ids=specific_ids, progress_bar=progress_bar,
                                         lut_entries=lut_entries)
        idt_masks = load_idt_masks(file=file, specific_ids=specific_ids, progress_bar=progress_bar)
        for ct_mask, idt_mask in zip(ct_masks, idt_masks):
            ct_mask[(idt_mask > 0) & (ct_mask == 0)] = 3
    return ct_masks


def load_tc_region_masks(file: Union[str, h5py.File],
                         specific_ids: Union[range, List[int], np.ndarray, None] = None,
                         progress_bar: bool = True) -> List[np.ndarray]:
    """Load Total Charge (TC) region masks as ground truth data.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Total Charge (TC) region masks
    """
    return [np.round(np.sum(occ, axis=-1)).astype(np.uint8) for occ in
            load_dataset(file=file, load_csds=False, load_occupations=True, specific_ids=specific_ids,
                         progress_bar=progress_bar).occupations]


def load_tc_region_minus_tct_masks(file: Union[str, h5py.File],
                                   specific_ids: Union[range, List[int], np.ndarray, None] = None,
                                   progress_bar: bool = True) -> List[np.ndarray]:
    """Load Total Charge (TC) region minus Total Charge Transition (TCT) masks as ground truth data (TCTs are basically excluded from the regions).

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Total Charge (TC) region minus Total Charge Transition (TCT) masks
    """
    return [np.round(np.sum(occ, axis=-1) - tct_mask).astype(np.uint8) for (occ, tct_mask) in
            zip(load_dataset(file=file, load_csds=False, load_occupations=True, specific_ids=specific_ids,
                             progress_bar=progress_bar).occupations,
                load_dataset(file=file, load_csds=False, load_tct_masks=True, specific_ids=specific_ids,
                             progress_bar=progress_bar).tct_masks)]


def load_c_region_masks(file: Union[str, h5py.File],
                        specific_ids: Union[range, List[int], np.ndarray, None] = None,
                        progress_bar: bool = True) -> List[np.ndarray]:
    """Load Charge (C) region masks as ground truth data (CTs are basically excluded from the TC regions).

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If you want to do multiple consecutive loads from the same file (e.g. for using th PyTorch
            SimcatsDataset without preloading), consider initializing the file object yourself and passing it, to
            improve the performance.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. Default is True.

    Returns:
        Charge (C) region masks
    """
    c_region_masks = load_tc_region_masks(file=file, specific_ids=specific_ids, progress_bar=progress_bar)
    ct_masks = load_ct_masks(file=file, specific_ids=specific_ids, progress_bar=progress_bar)
    for ct_mask, c_region_mask in zip(ct_masks, c_region_masks):
        c_region_mask[c_region_mask == ct_mask] = 0
    return c_region_masks
