"""
Some tools I've built which might be useful in the future as separate
functions.

"""

def calculateRegression(x, y, type):
    """
    Calculate three types of regression:
        1. Linear (lin)
        2. Log (log)
        3. Exponential (exp)
        4. Linear through zero (lin0)
    """

    try:
        import numpy as np
    except ImportError:
        raise ImportError('NumPy not found')

    try:
        from scipy import stats
    except ImportError:
        raise ImportError('SciPy not found')

    # Check the smallest value in x or y isn't below ~1x10-7 otherwise
    # we hit numerical instabilities.
    xFactorFix, xFactor = False, 0
    yFactorFix, yFactor = False, 0
    minX = min(x)
    minY = min(y)
    if minX < 2e-7:
        print 'Scaling x-data to improve numerical stability'
        x, xFactor, xFactorFix = fixRange(x)
    if minY < 2e-7:
        print 'Scaling y-data to improve numerical stability'
        y, yFactor, yFactorFix = fixRange(y)


    if type is 'lin0':
        xf = x
        x = x[:,np.newaxis] # make a singleton extra dimension
        m, _, _, _ = np.linalg.lstsq(x, y)
        c, r, p = 0, np.nan, np.nan
    elif type is 'lin':
        print 'lin'
        m, c, r, p, std_err = stats.linregress(x, y)
        xf = x
    elif type is 'log':
        m, c, r, p, std_err = stats.linregress(np.log10(x),y)
        xf = np.log10(x)
    elif type is 'exp':
        m, c, r, p, std_err = stats.linregress(x,np.log10(y))
        xf = x
    else:
        raise ValueError('Unknown regression type')

    yf = (m * xf) + c

    if xFactorFix:
        xf = xf / 10**xFactor
    if yFactorFix:
        yf = yf / 10**yFactor

    return xf, yf, m, c, r


def calculatePolyfit(x, y):
    """ Calculate a linear regression using polyfit instead """

    # FIXME(pica) This doesn't work for large ranges (tens of orders of
    # magnitude). No idea why as the fixRange() function above should make
    # all values greater than one. The trouble mainly seems to happen when
    # log10(min(x)) negative.

    try:
        from scipy import polyfit, polyval
    except ImportError:
        raise ImportError('SciPy not found')

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


def coefficientOfDetermination(obs, model):
    """ Calculate the coefficient of determination for a modelled function """
    try:
        import numpy as np
    except ImportError:
        raise ImportError('NumPy not found')

    obsBar = np.mean(obs)
    modelBar = np.mean(model)

    SStot = np.sum((obs - obsBar)**2)
    SSreg = np.sum((model - obsBar)**2)
    R2 = SSreg / SStot

    return R2
