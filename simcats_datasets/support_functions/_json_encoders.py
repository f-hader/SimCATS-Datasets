"""JSON encoders that are required to encode numpy arrays and xarray DataArrays while saving datasets.

@author: f.hader
"""

import json
import numpy as np
from xarray import DataArray


class MultipleJsonEncoders(json.JSONEncoder):
    """Combine multiple JSON encoders"""

    def __init__(self, *encoders):
        super().__init__()
        self.encoders = encoders
        self.args = ()
        self.kwargs = {}

    def default(self, obj):
        for encoder in self.encoders:
            try:
                return encoder(*self.args, **self.kwargs).default(obj)
            except TypeError:
                pass
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        enc = json.JSONEncoder(*args, **kwargs)
        enc.default = self.default
        return enc


class DataArrayEncoder(json.JSONEncoder):
    """JSON encoder for DataArrays."""

    def default(self, obj):
        if isinstance(obj, DataArray):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
