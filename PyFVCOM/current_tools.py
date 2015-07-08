"""
Tools to work with current data. Reuses some functions from tide_tools.

"""

import numpy as np


def scalar2vector(direction, speed):
    """
    Convert arrays of two scalars into the corresponding vector components.
    This is mainly meant to be used to convert direction and speed to the u and
    v velocity components.

    Parameters
    ----------
    direction, speed : ndarray
        Arrays of direction (degrees) and speed (any units).

    Returns
    -------
    u, v : ndarray
        Arrays of the u and v components of the speed and direction in units of
        speed.

    """

    u = np.sin(np.deg2rad(direction)) * speed
    v = np.cos(np.deg2rad(direction)) * speed

    return u, v


def vector2scalar(u, v):
    """
    Convert two vector components into the scalar values. Mainly used for
    converting u and v velocity components into direction and speed.

    Parameters
    ----------
    u, v : ndarray
        Arrays of (optionally time, space (vertical and horizontal) varying)
        u and v vectors.

    Returns
    -------
    direction, speed : ndarray
        Arrays of direction (degrees) and speed (u and v units).

    """

    direction = np.rad2deg(np.arctan2(u, v))
    speed = np.sqrt(u**2 + v**2)

    return direction, speed
