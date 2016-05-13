"""
Some tools I've built which might be useful in the future as separate
functions.

"""

import inspect

import numpy as np

from warnings import warn
from scipy import stats, polyfit, polyval


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


def coefficient_of_determination(obs, model):
    """ Calculate the coefficient of determination for a modelled function """
    obsBar = np.mean(obs)

    SStot = np.sum((obs - obsBar)**2)
    SSreg = np.sum((model - obsBar)**2)
    R2 = SSreg / SStot

    return R2


def fix_range(a, nmin, nmax):
    """
    Given an array of values `a', scale the values within in to the range
    specified by `nmin' and `nmax'.

    Parameters
    ----------
    a : ndarray
        Array of values to scale.
    nmin, nmax : float
        New minimum and maximum values for the new range.

    Returns
    -------
    b : ndarray
        Scaled array.

    """

    A = a.min()
    B = a.max()
    C = nmin
    D = nmax

    b = (((D - C) * (a - A)) / (B - A)) + C

    return b


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


# For backwards compatibility.
def calculateRegression(*args, **kwargs):
    warn('{} is deprecated. Use calculate_regression instead.'.format(inspect.stack()[0][3]))
    return calculate_regression(*args, **kwargs)


# For backwards compatibility.
def calculatePolyfit(*args, **kwargs):
    warn('{} is deprecated. Use calculate_polyfit instead.'.format(inspect.stack()[0][3]))
    return calculate_polyfit(*args, **kwargs)


# For backwards compatibility.
def coefficientOfDetermination(*args, **kwargs):
    warn('{} is deprecated. Use coefficient_of_determination instead.'.format(inspect.stack()[0][3]))
    return coefficient_of_determination(*args, **kwargs)


# For backwards compatibility.
def fixRange(*args, **kwargs):
    warn('{} is deprecated. Use fix_range instead.'.format(inspect.stack()[0][3]))
    return fix_range(*args, **kwargs)
