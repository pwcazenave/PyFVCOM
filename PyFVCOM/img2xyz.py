"""
Functions to convert images to values along a given set of vertical elevations.

"""

import colorsys

import numpy as np


def rgb2z(R, G, B, zlev, parm='H'):
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
    parm : str
        Which parameter to use from the HSV decomposition: 'H' (hue), 'S'
        (saturation) or 'V' (value). Default is 'H'. Hue works well for rainbow
        colour palettes and value works well for grayscale.

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
    ny, nx = np.shape(R)
    H = np.empty(R.shape)
    S = np.empty(R.shape)
    V = np.empty(R.shape)
    for xi, xx in enumerate(range(R.shape[0])):
        for yi, yy in enumerate(range(R.shape[1])):
            H[xi, yi], S[xi, yi], V[xi, yi] = colorsys.rgb_to_hsv(
                R[xi, yi], G[xi, yi], B[xi, yi])

    # Clear out the weird -1 values
    # H[H > 0.7] = H[H < 0.7].max()

    # Convert the scaling RGBs to hues.
    h, s, v = [], [], []
    for lev in zlev:
        rgb = np.asarray(lev[1:])
        th, ts, tv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        h.append(th)
        s.append(ts)
        v.append(tv)
    del th, ts, tv

    # Depending on the value of parm, create an array C which contains the
    # colour data we're translating to depths.
    if parm is 'H':
        C, ci = H, h
    elif parm is 'S':
        C, ci = S, s
    elif parm is 'V':
        C, ci = V, v

    # Now go through the associated depths (in pairs) and find the values
    # within C which fall between the corresponding h values and scale them to
    # the depths.
    z = np.zeros((ny, nx))  # images are all backwards
    nz = zlev.shape[0]
    for i in range(1, nz):

        cs = ci[i - 1]
        ce = ci[i]
        if ce < cs:
            ce, cs = cs, ce
        zs = zlev[i - 1, 0]
        ze = zlev[i, 0]
        if ze < zs:
            ze, zs = zs, ze

        # The range probably shouldn't be inclusive at both ends...
        idx = np.where((C >= cs) * (C <= ce))

        if C[idx].max() - C[idx].min() == 0:
            # If the range of colours in this level is zero, we've got discrete
            # colours (probably). As such, the scaling approach won't work, so
            # we'll just assign all the values we've found to the mean of the
            # two depths at the extremes of the current set of colours.
            z[idx] = (ze - zs) / 2.0

        else:
            zi = (((ze - zs) *
                   (C[idx] - C[idx].min())) /
                  (C[idx].max() - C[idx].min())) + zs

            z[idx] = zi

    return z
