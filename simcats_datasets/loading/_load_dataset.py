"""The main interface for loading simcats datasets, typically stored as HDF5 files.

@author: f.hader
"""

import json

from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy

from typing import List, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm


def load_dataset(file: Union[str, h5py.File],
                 load_csds=True,
                 load_occupations: bool = False,
                 load_tct_masks: bool = False,
                 load_ct_by_dot_masks: bool = False,
                 load_line_coords: bool = False,
                 load_line_labels: bool = False,
                 load_metadata: bool = False,
                 load_ids: bool = False,
                 specific_ids: Union[range, List[int], np.ndarray, None] = None,
                 progress_bar: bool = False) -> Tuple:
    """Loads a dataset consisting of multiple CSDs from a given path.

    Args:
        file: The file to read the data from. Can either be an object of the type `h5py.File` or the path to the
            dataset. If a path is supplied, load_dataset will open the file itself. If you want to do multiple
            consecutive loads from the same file (e.g. for using th PyTorch SimcatsDataset without preloading), consider
            initializing the file object yourself and passing it, to improve the performance.
        load_csds: Determines if csds should be loaded. Default is True.
        load_occupations: Determines if occupation data should be loaded. Default is False.
        load_tct_masks: Determines if lead transition masks should be loaded. Default is False.
        load_ct_by_dot_masks: Determines if charge transition labeled by affected dot masks should be loaded. This
            requires that ct_by_dot_masks have been added to the dataset. If a dataset has been created using
            create_simulated_dataset, these masks can be added afterwards using add_ct_by_dot_masks_to_dataset, mainly
            to avoid recalculating them multiple times (for example for machine learning purposes). Default is False.
        load_line_coords: Determines if lead transition definitions using start and end points should be loaded. Default
            is False.
        load_line_labels: Determines if labels for lead transitions defined using start and end points should be loaded.
            Default is False.
        load_metadata: Determines if the metadata (SimCATS config) of the CSDs should be loaded. Default is False.
        load_ids: Determines if the available ids should be loaded (or in case of specific ids: the specific ids are
            returned in the given order). Default is False.
        specific_ids: Determines if only specific ids should be loaded. Using this option, the returned values are
            sorted according to the specified ids and not necessarily ascending. If set to None, all data is loaded.
            Default is None.
        progress_bar: Determines whether to display a progress bar. This parameter has no functionality since version 2,
            but is kept for compatibility reasons. Default is False.

    Returns:
        namedtuple: The namedtuple can be unpacked like every normal tuple, or instead accessed by field names. \n
        Depending on what has been enabled, the following data is included in the named tuple: \n
        - field 'csds': List containing all CSDs as numpy arrays. The list is sorted by the id of the CSDs (if no
          specific_ids are provided, else the order is given by specific_ids).
        - field 'occupations': List containing numpy arrays with occupations.
        - field 'tct_masks': List containing numpy arrays of TCT masks.
        - field 'ct_by_dot_masks': List containing numpy arrays of CT_by_dot masks.
        - field 'line_coordinates': List containing numpy arrays of line coordinates.
        - field 'line_labels': List containing a list of dictionaries (one dict for each line specified as line
            coordinates).
        - field 'metadata': List containing dictionaries with all metadata (simcats configs) for each CSD.
        - field 'ids': List of the ids of the CSDs.
    """
    # fieldname are used for the namedtuple, to make fields accessible by names
    fieldnames = []
    if load_csds:
        fieldnames.append("csds")
    if load_occupations:
        fieldnames.append("occupations")
    if load_tct_masks:
        fieldnames.append("tct_masks")
    if load_ct_by_dot_masks:
        fieldnames.append("ct_by_dot_masks")
    if load_line_coords:
        fieldnames.append("line_coordinates")
    if load_line_labels:
        fieldnames.append("line_labels")
    if load_metadata:
        fieldnames.append("metadata")
    if load_ids:
        fieldnames.append("ids")
    CSDDataset = namedtuple(typename="CSDDataset", field_names=fieldnames)

    # use nullcontext to catch the case where a file is passed instead of the string
    with h5py.File(file, "r") if isinstance(file, str) else nullcontext(file) as _file:
        # if only specific ids should be loaded, check if all ids are available
        if specific_ids is not None:
            if isinstance(specific_ids, list) or isinstance(specific_ids, np.ndarray):
                # remember the previous order to undo the sorting that is required for reading from h5
                specific_ids = deepcopy(specific_ids)
                undo_sort_ids = np.argsort(np.argsort(specific_ids))
                specific_ids.sort()
            else:
                undo_sort_ids = None
        if load_ids:
            # only check if ids are correct, if load_ids is True. This prevents initializing a non-preloaded PyTorch
            # Dataset with non-existing specific IDs (which else would only crash as soon as a non-existent ID is
            # requested during training). We can't check this on loading CSDs etc. as it massively slows down loading.
            if specific_ids is not None:
                if np.min(specific_ids) < 0 or np.max(specific_ids) >= len(_file["csds"]):
                    msg = "Not all ids specified by 'specific_ids' are available in the dataset!"
                    raise IndexError(msg)
                available_ids = specific_ids
            else:
                available_ids = range(len(_file["csds"]))

        if load_csds:
            if specific_ids is not None:
                csds = _file["csds"][specific_ids]
            else:
                csds = _file["csds"][:]
        if load_occupations:
            if specific_ids is not None:
                occupations = _file["occupations"][specific_ids]
            else:
                occupations = _file["occupations"][:]
        if load_tct_masks:
            if specific_ids is not None:
                tct_masks = _file["tct_masks"][specific_ids]
            else:
                tct_masks = _file["tct_masks"][:]
        if load_ct_by_dot_masks:
            if specific_ids is not None:
                ct_by_dot_masks = _file["ct_by_dot_masks"][specific_ids]
            else:
                ct_by_dot_masks = _file["ct_by_dot_masks"][:]
        if load_line_coords:
            if specific_ids is not None:
                # remove padded nan values
                line_coords = [l_c[~np.isnan(l_c)].reshape((-1, 4)) for l_c in _file["line_coordinates"][specific_ids]]
            else:
                # remove padded nan values
                line_coords = [l_c[~np.isnan(l_c)].reshape((-1, 4)) for l_c in _file["line_coordinates"][:]]
        if load_line_labels:
            if specific_ids is not None:
                line_labels = [json.loads(l_l.tobytes().strip().decode("utf-8")) for l_l in
                               _file["line_labels"][specific_ids]]
            else:
                line_labels = [json.loads(l_l.tobytes().strip().decode("utf-8")) for l_l in _file["line_labels"][:]]
        if load_metadata:
            if specific_ids is not None:
                metadata = [json.loads(meta.tobytes().strip().decode("utf-8")) for meta in
                            _file["metadata"][specific_ids]]
            else:
                metadata = [json.loads(meta.tobytes().strip().decode("utf-8")) for meta in _file["metadata"][:]]

    # create a list of the further data to be returned (if activated)
    return_data = []
    if load_csds:
        return_data.append(csds)
    if load_occupations:
        return_data.append(occupations)
    if load_tct_masks:
        return_data.append(tct_masks)
    if load_ct_by_dot_masks:
        return_data.append(ct_by_dot_masks)
    if load_line_coords:
        return_data.append(line_coords)
    if load_line_labels:
        return_data.append(line_labels)
    if load_metadata:
        return_data.append(metadata)
    if load_ids:
        return_data.append(available_ids)

    # revert sorting if specific ids were used
    if specific_ids is not None and undo_sort_ids is not None:
        return_data = [[x[i] for i in undo_sort_ids] for x in return_data]

    return CSDDataset._make(tuple(return_data))
