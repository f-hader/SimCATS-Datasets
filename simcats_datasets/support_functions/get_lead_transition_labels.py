"""Functionalities for extracting labeled transition lines from a SimCATS CSD (using the metadata).

@author: f.hader
"""

from typing import Dict, List, Tuple

import numpy as np
from simcats.ideal_csd import IdealCSDInterface
from simcats.ideal_csd.geometric import calculate_all_bezier_anchors as calc_anchors

from simcats_datasets.support_functions.clip_line_to_rectangle import clip_point_line_to_rectangle, \
    clip_slope_line_to_rectangle, create_rectangle_corners


def get_lead_transition_labels(sweep_range_g1: np.ndarray,
                               sweep_range_g2: np.ndarray,
                               ideal_csd_config: IdealCSDInterface,
                               lead_transition_mask: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """Function for calculating the line coordinates and labels for all linear parts in a simulated CSD.

    **Warning**: This function expects that IdealCSDGeometric has been used for the simulation. Dot jumps or similar
    distortions are not taken into account in the calculation of the line coordinates. This means, that the returned
    lines are the ideal (undisturbed) lines.

    Args:
        sweep_range_g1: The sweep range for gate 1. Required to know where the boundaries are.
        sweep_range_g2: The sweep range for gate 2. Required to know where the boundaries are.
        ideal_csd_config: The IdealCSDInterface implementation that was used during the simulation. It is
            required to calculate the bezier anchors from the configured TCTs.
        lead_transition_mask: Lead transition mask (TCT mask), used to identify involved TCTs.

    Returns:
        np.ndarray, list[dict]: Array with the line coordinates and list containing dictionaries with corresponding
            labels. Every row of the array represents one line as [x_start, y_start, x_stop, y_stop].

    """
    # retrieve which TCTs are contained in the simulated csd
    tct_ids = np.unique(lead_transition_mask).astype(int).tolist()
    tct_ids.remove(0)

    # retrieve TCT rotation
    rotation = ideal_csd_config.rotation

    # get CSD corner point
    rect_corners = create_rectangle_corners(x_range=sweep_range_g1, y_range=sweep_range_g2)

    # list to collect labels
    line_points = []
    labels = []

    # for every tct find the linear parts that are included in the csd (to be used as labels for line detection)
    for i in tct_ids:
        # retrieve tct parameters
        tct_params = ideal_csd_config.tct_params[i - 1]
        # retrieve all bezier anchors. Linear parts are always bound by anchors of two subsequent triple points,
        # or by one anchor and infinte linear prolongation in the single dot regions.
        anchors = calc_anchors(tct_params=tct_params, max_peaks=i)

        # iterate all lead transitions / linear parts of the current tct and check if they are in the image
        for trans_id in range(i * 2):
            # the first lead transition only has one bezier anchor, as it is infinitively prolonged in the single dot
            # regime
            if trans_id == 0:
                anchor = anchors[i][trans_id, 0, :]
                slope = tct_params[2]
                # rotate slope into image space
                angle = np.arctan(slope) + rotation
                if slope < 0:
                    angle += np.pi
                slope = np.tan(angle)
                clipped_start, clipped_end = clip_slope_line_to_rectangle(slope=slope, point=anchor,
                                                                          rect_corners=rect_corners, is_start=False)
                if clipped_start is not None and clipped_end is not None:
                    line_points.append(np.array([clipped_start[0], clipped_start[1], clipped_end[0], clipped_end[1]]))
                    labels.append({"tct_id": i, "transition_id": trans_id})
            # the last lead transition only has one bezier anchor, as it is infinitively prolonged in the single dot
            # regime
            elif trans_id == i * 2 - 1:
                anchor = anchors[i][trans_id - 1, 2, :]
                slope = tct_params[3]
                # rotate slope into image space
                angle = np.arctan(slope) + rotation
                if slope < 0:
                    angle += np.pi
                slope = np.tan(angle)
                clipped_start, clipped_end = clip_slope_line_to_rectangle(slope=slope, point=anchor,
                                                                          rect_corners=rect_corners, is_start=True)
                if clipped_start is not None and clipped_end is not None:
                    line_points.append(np.array([clipped_start[0], clipped_start[1], clipped_end[0], clipped_end[1]]))
                    labels.append({"tct_id": i, "transition_id": trans_id})
            # all other transitions are in the double dot regime and have two anchors defining the line
            else:
                anchor_start = anchors[i][trans_id - 1, 2, :]
                anchor_stop = anchors[i][trans_id, 0, :]
                clipped_start, clipped_end = clip_point_line_to_rectangle(start=anchor_start, end=anchor_stop,
                                                                          rect_corners=rect_corners)
                if clipped_start is not None and clipped_end is not None:
                    line_points.append(np.array([clipped_start[0], clipped_start[1], clipped_end[0], clipped_end[1]]))
                    labels.append({"tct_id": i, "transition_id": trans_id})

    return np.array(line_points), labels
