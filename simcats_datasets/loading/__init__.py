"""Module with functionalities for loading data from dataset files (HDF5 format).

Also contains functionalities for loading data as pytorch dataset with different ground truth types.
"""

from simcats_datasets.loading._load_dataset import load_dataset

__all__ = ["load_dataset"]