"""Module with functions for creating a csd dataset using SimCATs for simulations.

@author: f.hader
"""

import itertools
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional

import h5py

# data handling
import numpy as np

# parallel
from parallelbar import progress_imap

# for SimCATS simulation
from simcats import Simulation, default_configs
from simcats.distortions import OccupationDotJumps
from simcats.support_functions import (
    LogNormalSamplingRange,
    NormalSamplingRange,
    UniformSamplingRange, ExponentialSamplingRange,
)
from tqdm import tqdm

from simcats_datasets.loading import load_dataset
from simcats_datasets.loading.load_ground_truth import load_ct_by_dot_masks
# label creation based on line intersection
from simcats_datasets.support_functions.get_lead_transition_labels import get_lead_transition_labels
from simcats_datasets.support_functions._json_encoders import NumpyEncoder

__all__ = []


def _simulate(args: Tuple) -> Tuple:
    """Method to simulate a csd with the given args. Required for parallel simulation in create_cimulated_dataset.

    Args:
        args: Tuple of sample_range_g1, sample_range_g2, volt_range, simcats_config, resolution.

    Returns:
        Tuple of csd, occ, lead_trans, metadata, line_points, labels.
    """
    sample_range_g1, sample_range_g2, volt_range, simcats_config, resolution = args

    # random number generator used for sampling volt ranges.
    # !Must be generated here! Else same for every process!
    rng = np.random.default_rng()
    # !also update the rng of the configs, because else all workers sample the same noise!
    for distortion in (
        *simcats_config["occupation_distortions"],
        *simcats_config["sensor_potential_distortions"],
        *simcats_config["sensor_response_distortions"],
    ):
        if hasattr(distortion, "rng"):
            distortion.rng = np.random.default_rng()
        if hasattr(distortion, "sigma"):
            # get sigma
            temp_sigma = distortion.sigma
            # modify sigma
            if isinstance(distortion.sigma, LogNormalSamplingRange):
                temp_sigma._LogNormalSamplingRange__rng = np.random.default_rng()
            elif isinstance(distortion.sigma, UniformSamplingRange):
                temp_sigma._UniformSamplingRange__rng = np.random.default_rng()
            elif isinstance(distortion.sigma, NormalSamplingRange):
                temp_sigma._NormalSamplingRange__rng = np.random.default_rng()
            elif isinstance(distortion.sigma, ExponentialSamplingRange):
                temp_sigma._ExponentialSamplingRange__rng = np.random.default_rng()
            # set sigma
            distortion.sigma = temp_sigma
    sim = Simulation(**simcats_config)

    # sample voltage ranges
    g1_start = rng.uniform(low=sample_range_g1[0], high=sample_range_g1[1])
    g2_start = rng.uniform(low=sample_range_g2[0], high=sample_range_g2[1])
    g1_range = np.array([g1_start, g1_start + volt_range[0]])
    g2_range = np.array([g2_start, g2_start + volt_range[1]])
    # perform simulation
    csd, occ, lead_trans, metadata = sim.measure(
        sweep_range_g1=g1_range, sweep_range_g2=g2_range, resolution=resolution
    )
    # calculate lead_transition labels
    ideal_csd_conf = metadata["ideal_csd_config"]
    line_points, labels = get_lead_transition_labels(
        sweep_range_g1=g1_range,
        sweep_range_g2=g2_range,
        ideal_csd_config=ideal_csd_conf,
        lead_transition_mask=lead_trans,
    )
    return csd, occ, lead_trans, metadata, line_points, labels


def create_simulated_dataset(
    dataset_path: str,
    simcats_config: dict = default_configs["GaAs_v1"],
    n_runs: int = 10000,
    resolution: np.ndarray = np.array([100, 100]),
    volt_range: np.ndarray = np.array([0.03, 0.03]),
    tags: Optional[dict] = None,
    num_workers: int = 1,
    progress_bar: bool = True,
    max_len_line_coordinates_chunk: int = 100,
    max_len_line_labels_chunk: int = 2000,
    max_len_metadata_chunk: int = 8000,
    dtype_csd: np.dtype = np.float32,
    dtype_occ: np.dtype = np.float32,
    dtype_tct: np.dtype = np.uint8,
    dtype_line_coordinates: np.dtype = np.float32,
) -> None:
    """Function for generating simulated datasets using SimCATS for simulations.

    **Warning**: This function expects that the simulation config uses IdealCSDGeometric from SimCATS. Other
    implementations are not guaranteed to work.

    Args:
        dataset_path: The path where the dataset will be stored. Can also be an already existing dataset, to which new
            data is added.
        simcats_config: Configuration for simcats simulation class. Default is the GaAs_v1 config provided by simcats.
        n_runs: Number of CSDs to be generated. Default is 10000.
        resolution: Pixel resolution for both axis of the CSDs, first number of columns (x), then number of rows (y).
            Default is np.array([100, 100]). \n
            Example: \n
            [res_g1, res_g2]
        volt_range: Volt range for both axis of the CSDs. Individual CSDs with the specified size are randomly sampled
            in the voltage space. Default is np.array([0.03, 0.03]) (usually the scans from RWTH GaAs offler sample are
            30mV x 30mV).
        tags: Additional tags for the data to be simulated, which will be added to the dataset DataFrame. Default is
            None. \n
            Example: \n
            {"tags": "shifted sensor, no noise", "sample": "GaAs"}.
        num_workers: Number of workers to parallelize dataset creation. Minimum is 1. Default is 1.
        progress_bar: Determines whether to display a progress bar. Default is True.
        max_len_line_coordinates_chunk: Maximum number of line coordinates. This is the size of the flattened array,
            therefore 100 means 20 lines. Default is 100.
        max_len_line_labels_chunk:  Maximum number of chars for the line label dict. Default is 2000.
        max_len_metadata_chunk: Maximum number of chars for the metadata dict. Default is 8000.
        dtype_csd: Specifies the dtype to be used for saving CSDs. Default is np.float32.
        dtype_occ: Specifies the dtype to be used for saving Occupations. Default is np.float32.
        dtype_tct: Specifies the dtype to be used for saving TCTs. Default is np.uint8.
        dtype_line_coordinates: Specifies the dtype to be used for saving line coordinates. Default is np.float32.
    """
    # set tags to an empty dict if none were supplied
    if tags is None:
        tags = {}

    # Create path where the dataset will be saved (if folder doesn't exist already)
    Path(Path(dataset_path).parent).mkdir(parents=True, exist_ok=True)

    # arange volt limits so that random sampling gives us a starting point that is at least the defined volt_range below
    # the maximum
    sample_range_g1 = simcats_config["volt_limits_g1"].copy()
    sample_range_g1[-1] -= volt_range[0]
    sample_range_g2 = simcats_config["volt_limits_g2"].copy()
    sample_range_g2[-1] -= volt_range[1]

    with h5py.File(dataset_path, "a") as hdf5_file:
        # load datasets or create them if not already there
        csds = hdf5_file.require_dataset(
            name="csds",
            shape=(0, resolution[1], resolution[0]),
            chunks=(1, resolution[1], resolution[0]),
            dtype=dtype_csd,
            maxshape=(None, resolution[1], resolution[0]),
        )
        occupations = hdf5_file.require_dataset(
            name="occupations",
            shape=(0, resolution[1], resolution[0], 2),
            chunks=(1, resolution[1], resolution[0], 2),
            dtype=dtype_occ,
            maxshape=(None, resolution[1], resolution[0], 2),
        )
        tct_masks = hdf5_file.require_dataset(
            name="tct_masks",
            shape=(0, resolution[1], resolution[0]),
            chunks=(1, resolution[1], resolution[0]),
            dtype=dtype_tct,
            maxshape=(None, resolution[1], resolution[0]),
        )
        line_coords = hdf5_file.require_dataset(
            name="line_coordinates",
            shape=(0, max_len_line_coordinates_chunk),
            chunks=(1, max_len_line_coordinates_chunk),
            dtype=dtype_line_coordinates,
            maxshape=(None, max_len_line_coordinates_chunk),
        )
        line_labels = hdf5_file.require_dataset(
            name="line_labels",
            shape=(0, max_len_line_labels_chunk),
            chunks=(1, max_len_line_labels_chunk),
            dtype=np.uint8,
            maxshape=(None, max_len_line_labels_chunk),
        )
        metadatas = hdf5_file.require_dataset(
            name="metadata",
            shape=(0, max_len_metadata_chunk),
            chunks=(1, max_len_metadata_chunk),
            dtype=np.uint8,
            maxshape=(None, max_len_metadata_chunk),
        )
        # determine index offset if there is already data in the dataset
        id_offset = csds.shape[0]

        # resize datasets to fit new data
        csds.resize(csds.shape[0] + n_runs, axis=0)
        occupations.resize(occupations.shape[0] + n_runs, axis=0)
        tct_masks.resize(tct_masks.shape[0] + n_runs, axis=0)
        line_coords.resize(line_coords.shape[0] + n_runs, axis=0)
        line_labels.resize(line_labels.shape[0] + n_runs, axis=0)
        metadatas.resize(metadatas.shape[0] + n_runs, axis=0)

        # simulate and save data
        indices = range(id_offset, n_runs + id_offset)
        arguments = itertools.repeat(
            (sample_range_g1, sample_range_g2, volt_range, simcats_config, resolution),
            times=len(indices),
        )
        for index, (csd, occ, lead_trans, metadata, line_points, labels) in zip(
            indices,
            progress_imap(
                func=_simulate,
                tasks=arguments,
                n_cpu=num_workers,
                total=len(indices),
                chunk_size=len(indices) // num_workers,
                disable=not progress_bar,
            ),
        ):
            # save data
            csds[index] = csd.astype(dtype_csd)
            occupations[index] = occ.astype(dtype_occ)
            tct_masks[index] = lead_trans.astype(dtype_tct)
            line_coords[index] = np.pad(
                line_points.flatten(),
                ((0, max_len_line_coordinates_chunk - line_points.size)),
                "constant",
                constant_values=np.nan,
            ).astype(dtype_line_coordinates)
            # Convert the line label dictionary to a JSON string
            json_line_labels = np.frombuffer(
                json.dumps(labels).encode("utf-8"), dtype=np.uint8
            )
            # pad with whitespace (" " in uint8 = 32) to a fixed size, so that we don't need the leaky special dtype
            json_line_labels_padded = np.pad(
                json_line_labels,
                ((0, max_len_line_labels_chunk - json_line_labels.size)),
                "constant",
                constant_values=32,
            )
            line_labels[index] = json_line_labels_padded

            # convert metadata
            metadata_converted = dict()
            for metadata_key, metadata_value in {**metadata, **tags}.items():
                if isinstance(metadata_value, np.ndarray):
                    metadata_converted[metadata_key] = metadata_value
                else:
                    metadata_converted[metadata_key] = str(metadata_value)
                    # add dot jumps to the metadata, to be able to apply it to all ground truth types
                    if metadata_key == "occupation_distortions":
                        for distortion in metadata_value:
                            if (
                                isinstance(distortion, OccupationDotJumps)
                                and distortion.activated
                            ):
                                metadata_converted[
                                    f"OccupationDotJumps_axis{distortion.axis}"
                                ] = distortion._OccupationDotJumps__previous_noise
            # save metadata
            metadata_converted = json.dumps(
                metadata_converted, cls=NumpyEncoder
            ).encode("utf-8")
            metadata_converted = np.frombuffer(metadata_converted, dtype=np.uint8)
            # pad with whitespace (" " in uint8 = 32) to a fixed size, so that we don't need the leaky special dtype
            metadata_padded = np.pad(
                metadata_converted,
                ((0, max_len_metadata_chunk - metadata_converted.size)),
                "constant",
                constant_values=32,
            )
            metadatas[index] = metadata_padded


def _load_ct_by_dot_masks_for_parallel(args: tuple) -> List[np.ndarray]:
    """Helper function for parallel loading of CT_by_dot masks in add_ct_by_dot_masks_to_dataset.

    Args:
        args: Tuple of arguments

    Returns:
        Loaded CT_by_dot masks
    """
    return load_ct_by_dot_masks(*args)


def add_ct_by_dot_masks_to_dataset(
    dataset_path: str,
    num_workers: int = 10,
    progress_bar: bool = True,
    dtype_ct_by_dot: np.dtype = np.uint8,
    batch_size_per_worker: int = 100,
) -> None:
    """Function for adding charge transitions labeled by dot masks to existing simulated datasets.

    Args:
        dataset_path: The path where the dataset is stored.
        num_workers: Number of workers to parallelize dataset creation. Minimum is 1. Default is 10.
        progress_bar: Determines whether to display a progress bar. Default is True.
        dtype_ct_by_dot: Specifies the dtype to be used for saving CT_by_dot masks. Default is np.uint8.
        batch_size_per_worker: Determines how many CT_by_dot masks are consecutively calculated by each worker, before
            saving them. Default is 100.
    """
    num_ids = len(load_dataset(file=dataset_path, load_csds=False, load_ids=True).ids)
    resolution = load_dataset(file=dataset_path, load_csds=True, specific_ids=[0]).csds[0].shape

    # setup id ranges for the batches
    id_ranges = list()
    for i in range(math.ceil(num_ids / batch_size_per_worker)):
        id_ranges.append(range(
            i * batch_size_per_worker, np.min([(i + 1) * batch_size_per_worker, num_ids])
        ))

    # Iterate and always calculate exactly one batch per worker, so that we can write to HDF5 after all workers have
    # finished their batch. SingleWriterMultipleReader mode of HDF5 was causing problems. Therefore, we now always write
    # after all workers have stopped and closed their file objects.
    with tqdm(unit="batches", total=len(id_ranges)) as pbar:
        for i in range(math.ceil(len(id_ranges) / num_workers)):
            temp_ct_by_dot_masks = list()
            temp_indices = list()

            arguments = zip(itertools.cycle([dataset_path]),
                            id_ranges[i*num_workers:(i+1)*num_workers],
                            itertools.cycle([False]),
                            itertools.cycle([1000]),
                            itertools.cycle([False])
                            )

            # calculate data
            for indices, _ct_by_dot_masks in zip(
                id_ranges[i*num_workers:(i+1)*num_workers],
                progress_imap(
                    func=_load_ct_by_dot_masks_for_parallel,
                    tasks=arguments,
                    n_cpu=num_workers,
                    total=len(id_ranges),
                    chunk_size=1,
                    disable=True,
                ),
            ):
                temp_ct_by_dot_masks.append(_ct_by_dot_masks)
                temp_indices.append(indices)
                pbar.update(1)

            # save data
            with h5py.File(dataset_path, "a") as hdf5_file:
                # load datasets or create them if not already there
                ct_by_dot_masks = hdf5_file.require_dataset(
                    name="ct_by_dot_masks",
                    shape=(num_ids, resolution[0], resolution[1]),
                    dtype=dtype_ct_by_dot,
                    chunks=(1, resolution[0], resolution[1]),
                    maxshape=(None, resolution[0], resolution[1]),
                )
                for ids, masks in zip(temp_indices, temp_ct_by_dot_masks):
                    if isinstance(masks, list):
                        masks = np.array(masks)
                    # just to be sure to save the masks one by one as chunks (as I absolutely don't trust HDF5 anymore)
                    for _id, _mask in zip(ids, masks):
                        ct_by_dot_masks[_id] = _mask.astype(dtype_ct_by_dot)
