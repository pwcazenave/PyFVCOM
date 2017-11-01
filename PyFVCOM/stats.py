"""
Some tools I've built which might be useful in the future as separate
functions.

"""

from __future__ import division

import inspect

import numpy as np

from warnings import warn
from scipy import stats, polyfit, polyval

from PyFVCOM.utilities import fix_range


def calculate_regression(x, y, type):
    """
    Calculate three types of regression:
        1. Linear (lin)
        2. Log (log)
        3. Exponential (exp)
        4. Linear through zero (lin0)
    """

    # Check the smallest value in x or y isn't below ~1x10-7 otherwise
    # we hit numerical instabilities.
    xFactorFix, xFactor = False, 0
    yFactorFix, yFactor = False, 0
    minX = min(x)
    minY = min(y)
    if minX < 2e-7:
        print('Scaling x-data to improve numerical stability')
        x, xFactor, xFactorFix = fixRange(x)
    if minY < 2e-7:
        print('Scaling y-data to improve numerical stability')
        y, yFactor, yFactorFix = fixRange(y)

    if type is 'lin0':
        xf = x
        x = x[:, np.newaxis]  # make a singleton extra dimension
        m, _, _, _ = np.linalg.lstsq(x, y)
        c, r, p = 0, np.nan, np.nan
    elif type is 'lin':
        print('lin')
        m, c, r, p, std_err = stats.linregress(x, y)
        xf = x
    elif type is 'log':
        m, c, r, p, std_err = stats.linregress(np.log10(x), y)
        xf = np.log10(x)
    elif type is 'exp':
        m, c, r, p, std_err = stats.linregress(x, np.log10(y))
        xf = x
    else:
        raise ValueError('Unknown regression type')

    yf = (m * xf) + c

    if xFactorFix:
        xf = xf / 10**xFactor
    if yFactorFix:
        yf = yf / 10**yFactor

    return xf, yf, m, c, r


def calculate_polyfit(x, y):
    """ Calculate a linear regression using polyfit instead """

    # FIXME(pica) This doesn't work for large ranges (tens of orders of
    # magnitude). No idea why as the fixRange() function above should make
    # all values greater than one. The trouble mainly seems to happen when
    # log10(min(x)) negative.

    # Check the smallest value in x or y isn't below 2x10-7 otherwise
    # we hit numerical instabilities.
    xFactorFix, xFactor = False, 0
    yFactorFix, yFactor = False, 0
    minX = min(x)
    minY = min(y)
    if minX < 2e-7:
        x, xFactor, xFactorFix = fixRange(x)
    if minY < 2e-7:
        y, yFactor, yFactorFix = fixRange(y)

    (ar, br) = polyfit(x, y, 1)
    yf = polyval([ar, br], x)
    xf = x

    if xFactorFix:
        xf = xf / 10**xFactor
    if yFactorFix:
        yf = yf / 10**yFactor

    return xf, yf


def rmse(a, b, axis=0):
    """
    Calculate the Root Mean Square Error (RMSE) between two identically sized
    arrays.

    RMSE = np.sqrt(np.mean((A - B)**2, axis=2))

    Parameters
    ----------
    a, b : ndarray
        Array of values to calculate RMSE.
    axis : int, optional
        Axis along which to calculate the mean. Defaults to the zeroth axis.

    Returns
    -------
    rmse: ndarray
        RMSE of `a' and `b'.

    """

    rmse = np.sqrt(np.mean((a - b)**2, axis=axis))

    return rmse


def calculate_coefficient(x, y, noisy=False):
    """
    Calculate the correlation coefficient and its p-value for two time series.

    Parameters
    ----------

    x, y : ndarray
        Time series arrays to correlate.
    noisy : bool, optional
        Set to True to enable verbose output (defaults to False).

    Returns
    -------

    r : ndarray
        Correlation coefficient.
    p : ndarray
        p-value for the corresponding correlation coefficient.

    Notes
    -----

    Using numpy.ma.corrcoef is ~5 slower than using scipy.stats.pearsonr
    despite giving the same results. In fact, the latter also gives the
    p-value.
    """

    # Timings for np.ma.corrcoef and scipy.stats.pearsonr:
    #   numpy:  738s
    #   scipy:  139s

    # Skip data with fewer than nine points.
    if len(np.ma.compressed(x)) > 9:
        #r = np.ma.corrcoef(xt, yt)[0, 1]
        r, p = stats.pearsonr(x, y)
    else:
        if noisy:
            print('Skipping data (all masked or fewer than 9 data points).')

    return r, p


# For backwards compatibility.
def calculateRegression(*args, **kwargs):
    warn('{} is deprecated. Use calculate_regression instead.'.format(inspect.stack()[0][3]))
    return calculate_regression(*args, **kwargs)


# For backwards compatibility.
def calculatePolyfit(*args, **kwargs):
    warn('{} is deprecated. Use calculate_polyfit instead.'.format(inspect.stack()[0][3]))
    return calculate_polyfit(*args, **kwargs)


# For backwards compatibility.
def fixRange(*args, **kwargs):
    warn('{} is deprecated. Use fix_range instead.'.format(inspect.stack()[0][3]))
    return fix_range(*args, **kwargs)
