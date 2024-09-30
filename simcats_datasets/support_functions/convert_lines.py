"""Helper functions for converting lines into different representations

@author: f.fuchs
"""

from copy import deepcopy
from typing import List, Union

import numpy as np



def lines_voltage_to_pixel_space(lines: Union[List[np.ndarray], np.ndarray],
                                 voltage_range_x: np.ndarray,
                                 voltage_range_y: np.ndarray,
                                 image_width: int,
                                 image_height: int,
                                 round_to_int: bool = False, ) -> np.ndarray:
    """Convert lines from voltage space to image/pixel space.
    This method makes a deepcopy of the supplied lines. Therefore, the original input won't be modified.

    Args:
        lines: Array or list of lines to convert, shape: (n, 4). \n
            Example: \n
            [[x_start, y_start, x_stop, y_stop], ...]
        voltage_range_x: Voltage range in x direction.
        voltage_range_y: Voltage range in y direction.
        image_width: Width of the image/pixel space.
        image_height: Height of the image/pixel space.
        round_to_int: Toggles if the lines are returned as floats (False) or are rounded and then returned as integers
            (True). Defaults to false.

    Returns:
        Array with rows containing the converted lines.
    """
    pixel_space = deepcopy(np.array(lines))
    for _i, line in enumerate(pixel_space):
        # change x coordinates of the line
        line[0] = (
                (image_width - 1) * (line[0] - voltage_range_x.min()) / (voltage_range_x.max() - voltage_range_x.min()))
        line[2] = (
                (image_width - 1) * (line[2] - voltage_range_x.min()) / (voltage_range_x.max() - voltage_range_x.min()))
        # change y coordinates of the line
        line[1] = ((image_height - 1) * (line[1] - voltage_range_y.min()) / (
                    voltage_range_y.max() - voltage_range_y.min()))
        line[3] = ((image_height - 1) * (line[3] - voltage_range_y.min()) / (
                    voltage_range_y.max() - voltage_range_y.min()))
    if round_to_int:
        return np.array(pixel_space).round(decimals=0).astype(int)
    else:
        return pixel_space


def lines_pixel_to_voltage_space(lines: Union[List[np.ndarray], np.ndarray],
                                 voltage_range_x: np.ndarray,
                                 voltage_range_y: np.ndarray,
                                 image_width: int,
                                 image_height: int, ) -> np.ndarray:
    """Convert lines from image/pixel space to voltage space.
    This method makes a deepcopy of the supplied lines. Therefore, the original input won't be modified.

    Args:
        lines: Array or list of lines to convert, shape: (n, 4). \n
            Example: \n
            [[x_start, y_start, x_stop, y_stop], ...]
        voltage_range_x: Voltage range in x direction.
        voltage_range_y: Voltage range in y direction.
        image_width: Width of the image/pixel space.
        image_height: Height of the image/pixel space.

    Returns:
        Array with rows containing the converted lines.
    """
    voltage_space = deepcopy(np.array(lines)).astype(np.float32)
    for _i, line in enumerate(voltage_space):
        # change x coordinates of the line
        line[0] = (line[0] / (image_width - 1)) * (voltage_range_x[1] - voltage_range_x[0]) + voltage_range_x[0]
        line[2] = (line[2] / (image_width - 1)) * (voltage_range_x[1] - voltage_range_x[0]) + voltage_range_x[0]
        # change y coordinates of the line
        line[1] = (line[1] / (image_height - 1)) * (voltage_range_y[1] - voltage_range_y[0]) + voltage_range_y[0]
        line[3] = (line[3] / (image_height - 1)) * (voltage_range_y[1] - voltage_range_y[0]) + voltage_range_y[0]
    return voltage_space


def lines_convert_two_coordinates_to_coordinate_plus_change(lines: Union[List[np.ndarray], np.ndarray]) -> np.ndarray:
    """Change the format from x,y,x,y to x,y,dx,dy.
    Order: top point > bottom point and if same y coordinate, right point > left point.

    Args:
        lines: Array or list of lines to convert, shape: (n, 4). \n
            Example: \n
            [[x_start, y_start, x_stop, y_stop], ...]

    Returns:
        Array with rows of lines in x,y,dx,dy format.
    """
    new_lines_pairs = []
    for line in lines:
        p1 = line[0], line[1]
        p2 = line[2], line[3]
        if p1[0] < p2[0]:
            new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
        elif p1[0] > p2[0]:
            new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
        else:
            if p1[1] < p2[1]:
                new_lines_pairs.append([p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1]])
            else:
                new_lines_pairs.append([p2[0], p2[1], p1[0] - p2[0], p1[1] - p2[1]])
    return np.array(new_lines_pairs)
