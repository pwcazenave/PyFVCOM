"""
Functions to convert images to values along a given set of vertical elevations.

"""

import colorsys
import os

import numpy as np
import matplotlib.pyplot as plt

def rgb2elev(R, G, B, zlev):
    """
    For the levels specified in zlev, interpolate the colour values in [R, G,
    B] which fall between each set of defined colour levels.

    Parameters
    ----------

    R, G, B : ndarray
        Image numpy array of the red, green and blue channels in the range
        0-255. Mask if you want to omit some parts from the scaling.
    zlev : ndarray
        Array of shape (z, 4) where the rows are elevation followed by an RGB
        triplet; z is the number of vertical levels to interpolate between.
        For best results, this should be as detailed as possible to reduce
        errors in the interpolation.

    Returns
    -------

    z : ndarray
        Array of values interpolated from the ranges specified in zlev.

    """

    if zlev.shape[-1] != 4:
        raise Exception('Specify [value, R, G, B] in the input zlev')

    # Make zlev floats (otherwise the conversion to HSV doesn't work.
    zlev = zlev.astype(float)

    # Some sort of bug in colorsys.rgb_to_hsv means we can't just dump the
    # arrays into the command like we can with rgb_to_yiq. Hence the horrible
    # for loop.
    H = np.empty(R.shape)
    S = np.empty(R.shape)
    V = np.empty(R.shape)
    for xi, xx in enumerate(xrange(R.shape[0])):
        for yi, yy in enumerate(xrange(R.shape[1])):
            H[xi, yi], S[xi, yi], V[xi, yi] = colorsys.rgb_to_hsv(R[xi, yi], G[xi, yi], B[xi, yi])

    # Clear out the weird -1 values
    H[H > 0.7] = H[H < 0.7].max()

    # Convert the RGBs to hues.
    h = []
    for lev in zlev:
        rgb = np.asarray(lev[1:])
        th, ts, tv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        h.append(th)
    del th, ts, tv

    # Now go through the associated depths (in pairs) and find the values
    # within H which fall between the corresponding h values and scale them to
    # the depths.
    z = np.zeros((ny, nx)) # images are all backwards
    nz = zlev.shape[0]
    for i in xrange(1, nz):

        hs = h[i -1]
        he = h[i]
        if he < hs:
            he, hs = hs, he
        zs = zlev[i -1, 0]
        ze = zlev[i, 0]
        if ze < zs:
            ze, zs = zs, ze

        idx = np.where((H >= hs) * (H < he))

        zi = (((ze - zs) * (H[idx] - H[idx].min())) / (H[idx].max() - H[idx].min())) + zs

        z[idx] = zi

    return z

