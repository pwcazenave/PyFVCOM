"""
Script to take a single channel tiff and, given its geographical extents,
convert it to ASCII xyz. Specify the range of values between which to
interpolate the depths.

"""

import sys

import numpy as np
import matplotlib.pyplot as plt


def colourInterp(R, G, B, zlev, fuzz=15):
    """
    For the levels specified in zlev, interpolate the colour values in im which
    fall between each set of levels.

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
    fuzz : int, optional
        Add a bit of fuzziness to the search for the search ranges. This will
        cause overlap between adjacent classes, but can be useful to fully
        capture all the area of interest.

    Returns
    -------

    elev : ndarray
        Array of values interpolated from the ranges specified in zlev.

    """

    if zlev.shape[-1] != 4:
        raise Exception('Specify [value, R, G, B] in the input zlev')

    nz = zlev.shape[0]

    elev = np.zeros((ny, nx))

    for i in xrange(1, nz):

        # Get the first set of ranges
        rgbs = zlev[i - 1, 1:]
        rgbe = zlev[i, 1:]
        zs = zlev[i -1, 0]
        ze = zlev[i, 0]

        #fuzz = 1 / (abs(ze - zs) / 50.0)
        fuzz = abs(ze - zs) * 2

        # To find the values which fall within the specified ranges, we need to
        # find the upper and lower values. We can't just assume the end value
        # (rgbe) is higher as the end value can be lower than the start value.
        rr = [rgbs[0], rgbe[0]]
        gg = [rgbs[1], rgbe[1]]
        bb = [rgbs[2], rgbe[2]]
        rs = min(rr) - fuzz
        re = max(rr) + fuzz
        gs = min(gg) - fuzz
        ge = max(gg) + fuzz
        bs = min(bb) - fuzz
        be = max(bb) + fuzz
        rimin = np.argmin(rr)
        gimin = np.argmin(rr)
        bimin = np.argmin(rr)

        if rs == re:
            re = re + 1

        if gs == ge:
            ge = ge + 1

        if bs == be:
            be = be + 1

        # Get the ranges and find which is the largest
        rgbi = np.array(((re - rs, ge - gs, be - bs))).argmax()

        # Find the parts of the image which fall within those ranges
        idx = np.where((R >= rs) * (R < re) * (G >= gs) * (G < ge) * (B >= bs) * (B < be))

        # Interpolate our z values to the intervals we've got from the RGB
        # triplets.
        if rgbi == 0:
            # Red
            print 'red',
            Rt = R[idx]
            if rimin == 0:
                print 'right'
                # Right way around
                zi = (((ze - zs) * (Rt - Rt.min())) / (Rt.max() - Rt.min())) + zs
            else:
                # Wrong way around
                print 'wrong'
                zi = (((zs - ze) * (Rt - Rt.min())) / (Rt.max() - Rt.min())) + ze
        elif rgbi == 1:
            # Green
            print 'green',
            Gt = G[idx]
            if gimin == 1:
                # Right way around
                print 'right'
                zi = (((ze - zs) * (Gt - Gt.min())) / (Gt.max() - Gt.min())) + zs
            else:
                # Wrong way around
                print 'wrong'
                zi = (((zs - ze) * (Gt - Gt.min())) / (Gt.max() - Gt.min())) + ze
        elif rgbi == 2:
            # Blue
            print 'blue',
            Bt = B[idx]
            if bimin == 1:
                # Right way around
                print 'right'
                zi = (((ze - zs) * (Bt - Bt.min())) / (Bt.max() - Bt.min())) + zs
            else:
                # Wrong way around
                print 'wrong'
                zi = (((zs - ze) * (Bt - Bt.min())) / (Bt.max() - Bt.min())) + ze

        elev[idx] = zi

    return elev


if __name__ == '__main__':

    # Image file first
    #tif = sys.argv[1]
    tif = 'xia_et_al_2010_fig6_clean_alpha.png'
    # Then coordinates as [west, east, south, north].
    #wesn = sys.argv[2]
    wesn = [366438.826916, 561926.062375, 5643743.000870, 5754525.445892]

    im = plt.imread(tif)

    # Images store their coordinates backwards.
    nx = im.shape[1]
    ny = im.shape[0]
    west, east, south, north = wesn
    xr = east - west
    yr = north - south
    x = np.arange(west, east, xr / nx)
    y = np.arange(south, north, yr / ny)

    # Now comes the hard bit. For the levels specified in zlev,
    zlev = np.array([[-52, 232,   8,   7],
            [-50, 255, 168,   0],
            [-45, 255, 255,   4],
            [-40, 188, 255,   7],
            [-35,  93, 255,   8],
            [-30,   0, 255,   0],
            [-25,   1, 252,  84],
            [-20,   0, 253, 167],
            [-15,   0, 255, 255],
            [-10,   0, 183, 238],
            [ -5,   5,  90, 255],
            [ -2,   0,  42, 255],
            [  0,   0,   0, 250]])
    zlev = np.array([[  0,   0,   0, 250],
        [ -2,   0,  42, 255],
        [ -5,   5,  90, 255],
        [-10,   0, 183, 238],
        [-15,   0, 255, 255],
        [-20,   0, 253, 167],
        [-25,   1, 252,  84],
        [-30,   0, 255,   0],
        [-35,  93, 255,   8],
        [-40, 188, 255,   7],
        [-45, 255, 255,   4],
        [-50, 255, 168,   0],
        [-52, 232,   8,   7]])

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    if R.max() <= 1:
        R = R * 255

    if G.max() <= 1:
        G = G * 255

    if B.max() <= 1:
        B = B * 255

    #elev = colourInterp(R, G, B, zlev, fuzz=15)
    Y, I, Q = colorsys.rgb_to_yiq(R, G, B)

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

    # Hue seems to vary quite nicely with the rainbow colour palette. We'll use
    # that the map our depth values.
    zs = zlev[:, 0].min()
    ze = zlev[:, 0].max()
    elev = (((ze - zs) * (H - H.min())) / (H.max() - H.min())) + zs


    # Add the alpha channel as a mask.
    if im.shape[2] == 4:
        mask = im[:, :, 3]
        # Make the mask binary (no graded boundaries).
        mask[mask > 0] = 1
        z = np.ma.array(elev, mask=False)
        z.mask[mask == 0] = True


    plt.figure()
    plt.imshow(z, cmap=plt.cm.jet_r)
    plt.colorbar()
    plt.clim(5, -65)
