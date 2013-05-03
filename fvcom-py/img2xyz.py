"""
Script to take a single channel tiff and, given its geographical extents,
convert it to ASCII xyz. Specify the range of values between which to
interpolate the depths.

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


if __name__ == '__main__':

    # Image file
    img = '/users/modellers/pica/Desktop/xia_et_al_2010_fig6_clean_alpha.png'
    # Then coordinates as [west, east, south, north].
    wesn = [366438.826916, 561926.062375, 5643743.000870, 5754525.445892]

    plotFig = False # plot an image of the converted elevations?

    im = plt.imread(img)

    # Images store their coordinates backwards.
    nx = im.shape[1]
    ny = im.shape[0]
    west, east, south, north = wesn
    xr = east - west
    yr = north - south
    x = np.arange(west, east, xr / nx)
    y = np.arange(south, north, yr / ny)

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    if R.max() <= 1:
        R = R * 255

    if G.max() <= 1:
        G = G * 255

    if B.max() <= 1:
        B = B * 255

    # Array of elevation values with the corresponding RGB colour triplet for
    # that depth. Colours will be converted to hue which is used as the key to
    # find the relevant pixels in the image array. Specify more levels to get
    # a better result here.
    zlev = np.array([[   2,   0,   0, 250],
        [  0,   0,  42, 255],
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
        [-52, 232,   8,   7]], dtype=float)

    # Map the specified elevations onto the image data.
    elev = rgb2elev(R, G, B, zlev)

    # Add the alpha channel as a mask.
    if im.shape[2] == 4:
        mask = im[:, :, 3]
        # Make the mask binary (no graded boundaries).
        mask[mask > 0] = 1
        z = np.ma.array(elev, mask=False)
        z.mask[mask == 0] = True
    else:
        z = np.ma.array(elev, mask=False)

    # Write out the coordinates as a CSV file.
    X, Y = np.meshgrid(x, y)
    xx = X.ravel()
    yy = Y.ravel()
    zz = np.flipud(z).ravel() # images are stored upside down

    csv = os.path.splitext(img)[0] + '.csv'
    np.savetxt(csv, np.transpose((xx, yy, zz)), delimiter=",", fmt='%.2f,%.2f,%.2f')

    if plotFig:
        plt.figure()
        plt.imshow(-z)
        plt.colorbar()
        plt.clim(-7, 65)
