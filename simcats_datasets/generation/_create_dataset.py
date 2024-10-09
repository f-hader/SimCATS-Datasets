"""Module with functions for creating a dataset from already existing data.

@author: f.hader
"""

import json
import h5py
from typing import Optional, List
from pathlib import Path
from os.path import dirname
import numpy as np

from simcats_datasets.support_functions._json_encoders import MultipleJsonEncoders, NumpyEncoder, DataArrayEncoder

__all__ = []


def create_dataset(dataset_path: str,
                   csds: Optional[List[np.ndarray]] = None,
                   sensor_scans: Optional[List[np.ndarray]] = None,
                   occupations: Optional[List[np.ndarray]] = None,
                   tct_masks: Optional[List[np.ndarray]] = None,
                   ct_by_dot_masks: Optional[List[np.ndarray]] = None,
                   line_coordinates: Optional[List[np.ndarray]] = None,
                   line_labels: Optional[List[dict]] = None,
                   metadata: Optional[List[dict]] = None,
                   max_len_line_coordinates_chunk: Optional[int] = None,
                   max_len_line_labels_chunk: Optional[int] = None,
                   max_len_metadata_chunk: Optional[int] = None,
                   dtype_csd: np.dtype = np.float32,
                   dtype_sensor_scan: np.dtype = np.float32,
                   dtype_occ: np.dtype = np.float32,
                   dtype_tct: np.dtype = np.uint8,
                   dtype_ct_by_dot: np.dtype = np.uint8,
                   dtype_line_coordinates: np.dtype = np.float32) -> None:
    """Function for creating simcats_datasets v2 format datasets from given data.

    Args:
        dataset_path: The path where the new (v2) HDF5 dataset will be stored.
        csds: The list of CSDs to use for creating the dataset. A dataset can have either CSDs or sensor scans, but
            never both. Default is None.
        sensor_scans: The list of sensor scans to use for creating the dataset. A dataset can have either CSDs or sensor
            scans, but never both. Default is None.
        occupations: List of occupations to use for creating the dataset. Defaults to None.
        tct_masks: List of TCT masks to use for creating the dataset. Defaults to None.
        ct_by_dot_masks: List of CT by dot masks to use for creating the dataset. Defaults to None.
        line_coordinates: List of line coordinates to use for creating the dataset. Defaults to None.
        line_labels: List of line labels to use for creating the dataset. Defaults to None.
        metadata: List of metadata to use for creating the dataset. Defaults to None.
        max_len_line_coordinates_chunk: The expected maximal length for line coordinates in number of float values (each
            line requires 4 floats). If None, it is set to the largest value of the CSD (or sensor scan) shape. Default
            is None.
        max_len_line_labels_chunk: The expected maximal length for line labels in number of uint8/char values (each line
            label, encoded as utf-8 json, should require at most 80 chars). If None, it is set to the largest value of
            the CSD (or sensor scan) shape * 20 (matching with allowed number of line coords). Default is None.
        max_len_metadata_chunk: The expected maximal length for metadata in number of uint8/char values (each metadata
            dict, encoded as utf-8 json, should require at most 8000 chars, expected rather something like 4000, but
            could get larger for dot jumps metadata of high resolution scans). If None, it is set to 8000. Default is
            None.
        dtype_csd: Specifies the dtype to be used for saving CSDs. Default is np.float32.
        dtype_sensor_scan: Specifies the dtype to be used for saving sensor scans. Default is np.float32.
        dtype_occ: Specifies the dtype to be used for saving Occupations. Default is np.float32.
        dtype_tct: Specifies the dtype to be used for saving TCTs. Default is np.uint8.
        dtype_ct_by_dot: Specifies the dtype to be used for saving CT by dot masks. Default is np.uint8.
        dtype_line_coordinates: Specifies the dtype to be used for saving line coordinates. Default is np.float32.
    """
    # Create path where the dataset will be saved (if folder doesn't exist already)
    Path(dirname(dataset_path)).mkdir(parents=True, exist_ok=True)

    # check if the dataset to be created is a csd or sensor_scan dataset
    if csds is not None and sensor_scans is None:
        csd_dataset = True
    elif csds is None and sensor_scans is not None:
        csd_dataset = False
    else:
        raise ValueError("A dataset can contain either CSDs or sensor scans but never both! Exactly one of the two has "
                         "to be None.")

    with h5py.File(dataset_path, "a") as hdf5_file:
        # get the number of total ids. This is especially required if a large dataset is loaded and saved step by step
        if csd_dataset:
            num_ids = len(csds)
        else:
            num_ids = len(sensor_scans)

        # get a temp copy of a csd or sensor scan (to get the shape) and retrieve the corresponding HDF5 dataset
        if csd_dataset:
            # process CSDs
            # save an example CSD to get shape and dtype
            temp_data = csds[0].copy()
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='csds',
                                           shape=(0, *temp_data.shape),
                                           dtype=dtype_csd,
                                           maxshape=(None, *temp_data.shape))
        else:
            # process sensor scans
            # save an example sensor scan to get shape and dtype
            temp_data = sensor_scans[0].copy()
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='sensor_scans',
                                           shape=(0, *temp_data.shape),
                                           dtype=dtype_sensor_scan,
                                           maxshape=(None, *temp_data.shape))
        # determine index offset if there is already data in the dataset
        id_offset = ds.shape[0]
        # resize datasets to fit new data
        ds.resize(ds.shape[0] + num_ids, axis=0)
        # Add new CSDs or sensor scans to the dataset
        if csd_dataset:
            ds[id_offset:] = np.array(csds).astype(dtype_csd)
        else:
            ds[id_offset:] = np.array(sensor_scans).astype(dtype_sensor_scan)
        if occupations is not None:
            if len(occupations) != num_ids:
                raise ValueError(
                    f"Number of new occupation arrays ({len(occupations)}) does not match the number of new CSDs or "
                    f"sensor scans ({num_ids}).")
            # process Occupations
            # save an example occ to get shape
            temp_occ = occupations[0].copy()
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='occupations', shape=(0, *temp_occ.shape), dtype=dtype_occ,
                                           maxshape=(None, *temp_occ.shape))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored occupation arrays ({ds.shape[0]}) does not match the number of already "
                    f"stored CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            ds[id_offset:] = np.array(occupations).astype(dtype_occ)
        if tct_masks is not None:
            if len(tct_masks) != num_ids:
                raise ValueError(
                    f"Number of new TCT mask arrays ({len(tct_masks)}) does not match the number of new CSDs or sensor "
                    f"scans ({num_ids}).")
            # process tct masks
            # save an example tct to get shape and dtype
            temp_tct = tct_masks[0].copy()
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='tct_masks', shape=(0, *temp_tct.shape), dtype=dtype_tct,
                                           maxshape=(None, *temp_tct.shape))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored TCT mask arrays ({ds.shape[0]}) does not match the number of already "
                    f"stored CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            ds[id_offset:] = np.array(tct_masks).astype(dtype_tct)
        if ct_by_dot_masks is not None:
            if len(ct_by_dot_masks) != num_ids:
                raise ValueError(
                    f"Number of new CT by dot mask arrays ({len(ct_by_dot_masks)}) does not match the number of new "
                    f"CSDs or sensor scans ({num_ids}).")
            # process tct masks
            # save an example tct to get shape and dtype
            temp_ct_by_dot = ct_by_dot_masks[0].copy()
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='ct_by_dot_masks', shape=(0, *temp_ct_by_dot.shape),
                                           dtype=dtype_ct_by_dot, maxshape=(None, *temp_ct_by_dot.shape))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored CT by dot mask arrays ({ds.shape[0]}) does not match the number of "
                    f"already stored CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            ds[id_offset:] = np.array(ct_by_dot_masks).astype(dtype_tct)
        if line_coordinates is not None:
            if len(line_coordinates) != num_ids:
                raise ValueError(
                    f"Number of new line coordinates ({len(line_coordinates)}) does not match the number of new "
                    f"CSDs or sensor scans ({num_ids}).")
            # retrieve fixed length for chunks
            if max_len_line_coordinates_chunk is None:
                # calculate max expected length (max_number_of_lines * 4 entries, max number estimated as max(shape)/4)
                max_len = max(temp_data.shape)
            else:
                max_len = max_len_line_coordinates_chunk
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='line_coordinates', shape=(0, max_len), dtype=dtype_line_coordinates,
                                           maxshape=(None, max_len))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored line coordinates ({ds.shape[0]}) does not match the number of already "
                    f"stored CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            # process line coordinates
            # pad to a fixed size, so that we don't need the leaky special dtype
            line_coordinates = np.array(
                [np.pad(l_c.flatten(), ((0, max_len - l_c.size)), 'constant', constant_values=np.nan) for l_c in
                 line_coordinates])
            ds[id_offset:] = line_coordinates.astype(dtype_line_coordinates)
        if line_labels is not None:
            if len(line_labels) != num_ids:
                raise ValueError(
                    f"Number of new line labels ({len(line_labels)}) does not match the number of new CSDs or sensor "
                    f"scans ({num_ids}).")
            # retrieve fixed length for chunks
            if max_len_line_labels_chunk is None:
                # calculate max expected length (max_number_of_lines * 80 uint8 numbers, max number estimated as
                # max(shape)/4)
                max_len = max(temp_data.shape) * 20
            else:
                max_len = max_len_line_labels_chunk
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='line_labels', shape=(0, max_len), dtype=np.uint8,
                                           maxshape=(None, max_len))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored line labels ({ds.shape[0]}) does not match the number of already stored "
                    f"CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            # process line labels
            line_labels = [json.dumps(l_l).encode("utf-8") for l_l in line_labels]
            # go to bytes array for better saving and loading
            line_labels = [np.frombuffer(l_l, dtype=np.uint8) for l_l in line_labels]
            # pad with whitespace (" " in uint8 = 32) to a fixed size, so that we don't need the leaky special dtype
            line_labels = np.array(
                [np.pad(l_l, ((0, max_len - l_l.size)), 'constant', constant_values=32) for l_l in line_labels])
            ds[id_offset:] = line_labels
        if metadata is not None:
            if len(metadata) != num_ids:
                raise ValueError(
                    f"Number of new metadata ({len(metadata)}) does not match the number of new CSDs or sensor scans "
                    f"({num_ids}).")
            # retrieve fixed length for chunks
            if max_len_metadata_chunk is None:
                # set len to 8000 uint8 numbers, that should already include some extra safety (expected smth. like
                # 3200-4000 chars)
                max_len = 8000
            else:
                max_len = max_len_metadata_chunk
            # use chunks as this will speed up reading later! One chunk is set to be exactly one image (optimized to
            # load one image at a time during training)
            ds = hdf5_file.require_dataset(name='metadata', shape=(0, max_len), dtype=np.uint8,
                                           maxshape=(None, max_len))
            if ds.shape[0] != id_offset:
                raise ValueError(
                    f"Number of already stored metadata ({ds.shape[0]}) does not match the number of already stored "
                    f"CSDs or sensor scans ({id_offset}).")
            # resize datasets to fit new data
            ds.resize(ds.shape[0] + num_ids, axis=0)
            # process metadata
            metadata = [json.dumps(meta, cls=MultipleJsonEncoders(NumpyEncoder, DataArrayEncoder)).encode("utf-8") for
                        meta in metadata]
            # go to bytes array for better saving and loading
            metadata = [np.frombuffer(m, dtype=np.uint8) for m in metadata]
            # pad with whitespace (" " in uint8 = 32) to a fixed size, so that we don't need the leaky special dtype
            metadata = np.array([np.pad(m, ((0, max_len - m.size)), 'constant', constant_values=32) for m in metadata])
            ds[id_offset:] = metadata
