"""Helper functions for clipping lines into rectangles (into the CSD space).

Used to clip single transition lines into the CSD space to generate transition specific labels with the function
`get_lead_transition_labels` from `simcats_datasets.support_functions.get_lead_transition_labels`.

@author: f.hader
"""

from typing import Tuple, List, Union


def clip_slope_line_to_rectangle(slope: float, point: Tuple[float, float], rect_corners: List[Tuple[float, float]],
                                 is_start: bool = True) -> Union[Tuple[Tuple[float, float], Tuple[float, float]], None]:
    """Clips a line segment with a given slope to a rectangle, extending it to either positive or negative infinity from the provided point.

    Args:
        slope: The slope of the line.
        point: A tuple (x, y) representing the starting or ending point of the line segment.
        rect_corners: A list of four tuples, each representing the corner points of the rectangle.
        is_start: Specifies whether the line extends to positive infinity (True) or negative infinity (False) from the
            provided point. Default is True (positive infinity).

    Returns:
        A tuple (start, end) representing the clipped line segment. Returns None if the line is entirely outside the
        rectangle.

    Notes:
        - The function handles lines defined by a slope and a single point.
        - The 'is_start' parameter determines whether the line extends to positive or negative infinity from the
          provided point.
    """
    x_range = (min(rect_corners, key=lambda p: p[0])[0], max(rect_corners, key=lambda p: p[0])[0])

    if is_start:
        start = point
        end = (x_range[1] + 1, point[1] + slope * (x_range[1] + 1 - point[0]))
    else:
        start = (x_range[0] - 1, point[1] - slope * (point[0] - x_range[0] + 1))
        end = point

    # Use the clip_line_to_rectangle function to clip the line
    return clip_point_line_to_rectangle(start, end, rect_corners)


def clip_infinite_slope_line_to_rectangle(slope: float, point: Tuple[float, float],
                                          rect_corners: List[Tuple[float, float]]) -> Union[
    Tuple[Tuple[float, float], Tuple[float, float]], None]:
    """Clips a line segment with a given slope to a rectangle, extending it to positive and negative infinity from the provided point.

    Args:
        slope: The slope of the line.
        point: A tuple (x, y) representing the starting or ending point of the line segment.
        rect_corners: A list of four tuples, each representing the corner points of the rectangle.

    Returns:
        A tuple (start, end) representing the clipped line segment. Returns None if the line is entirely outside the
        rectangle.

    Notes:
        - The function handles lines defined by a slope and a single point.
    """
    x_range = (min(rect_corners, key=lambda p: p[0])[0], max(rect_corners, key=lambda p: p[0])[0])

    start = (x_range[0] - 1, point[1] - slope * (point[0] - x_range[0] + 1))
    end = (x_range[1] + 1, point[1] + slope * (x_range[1] + 1 - point[0]))

    # Use the clip_line_to_rectangle function to clip the line
    return clip_point_line_to_rectangle(start, end, rect_corners)


def clip_point_line_to_rectangle(start: Tuple[float, float], end: Tuple[float, float],
                                 rect_corners: List[Tuple[float, float]]) -> Union[
    Tuple[Tuple[float, float], Tuple[float, float]], None]:
    """Clips a line segment defined by its start and end points to a rectangle.

    Args:
        start: A tuple (x, y) representing the start point of the line.
        end: A tuple (x, y) representing the end point of the line.
        rect_corners: A list of four tuples, each representing the corner points of the rectangle.

    Returns:
        A tuple representing the clipped line segment (start, end) if any part of the line is inside the rectangle.
        Returns None if the line is entirely outside the rectangle.

    Notes:
        - The function handles lines defined by two points.
        - The function handles the case when the line is entirely inside the rectangle.
    """
    if all(is_point_inside_rectangle(point, rect_corners) for point in (start, end)):
        # The entire line is inside the rectangle
        return start, end

    clipped_start = None
    clipped_end = None

    # Check if the start point is inside the rectangle
    if is_point_inside_rectangle(start, rect_corners):
        clipped_start = start

    # Check if the end point is inside the rectangle
    if is_point_inside_rectangle(end, rect_corners):
        clipped_end = end

    # Iterate through pairs of adjacent corner points to check for intersections
    for rect_point1, rect_point2 in zip(rect_corners, rect_corners[1:] + [rect_corners[0]]):
        # Calculate the intersection point between the line and the rectangle edge
        intersection = line_intersection(start, end, rect_point1, rect_point2)
        if intersection is not None:
            if clipped_start is None:
                clipped_start = intersection
            elif clipped_end is None:
                clipped_end = intersection
            if clipped_start is not None and clipped_end is not None:
                break

    return clipped_start, clipped_end


def is_point_inside_rectangle(point: Tuple[float, float], rect_corners: List[Tuple[float, float]]) -> bool:
    """Checks if a point is inside a rectangle defined by its corner points.

    Args:
        point: A tuple (x, y) representing the point to be checked.
        rect_corners: A list of four tuples, each representing the corner points of the rectangle. it is assumed that
            these are sorted so that they from a course around the rectangle. Thus, the first and third (or
            alternatively the second and fourth) define the rectangle.

    Returns:
        True if the point is inside the rectangle, False otherwise.
    """
    x, y = point
    x1, y1 = rect_corners[0]
    x2, y2 = rect_corners[2]
    return x1 <= x <= x2 and y1 <= y <= y2


def line_intersection(p1: Tuple[float, float], p2: Tuple[float, float], q1: Tuple[float, float],
                      q2: Tuple[float, float]) -> Union[Tuple[float, float], None]:
    """Calculates the intersection point between two line segments defined by their respective endpoints.

    Args:
        p1: A tuple (x, y) representing the first endpoint of the first line.
        p2: A tuple (x, y) representing the second endpoint of the first line.
        q1: A tuple (x, y) representing the first endpoint of the second line.
        q2: A tuple (x, y) representing the second endpoint of the second line.

    Returns:
        A tuple (x, y) representing the intersection point if the lines intersect. Returns None if the lines are
        parallel or do not intersect.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2

    # denominator
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Lines are parallel or coincident

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        intersection_x = x1 + t * (x2 - x1)
        intersection_y = y1 + t * (y2 - y1)
        return (intersection_x, intersection_y)
    else:
        return None


def create_rectangle_corners(x_range: Tuple[float, float], y_range: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Creates rectangle corner points that form a rectangle around the specified x and y value ranges.

    Args:
        x_range: A tuple (x_min, x_max) representing the minimum and maximum x values.
        y_range: A tuple (y_min, y_max) representing the minimum and maximum y values.

    Returns:
        A list of four tuples, each representing the corner points of the rectangle in the following order: bottom-left,
        bottom-right, top-right, top-left.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    bottom_left = (x_min, y_min)
    bottom_right = (x_max, y_min)
    top_right = (x_max, y_max)
    top_left = (x_min, y_max)

    return [bottom_left, bottom_right, top_right, top_left]
