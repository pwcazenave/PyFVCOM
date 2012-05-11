#!/usr/bin/env python

def fixRange(data):
    """ Reduces the minimum so that the calculations are numerically stable """
    
    from numpy import log10, min, max, abs

    dataFactor = abs(log10(min(data)))
    scaled = data * 10**dataFactor

    fixed = True

    return scaled, dataFactor, fixed


def calculateRegression(x, y, type):
    """ Calculate three types of regression:
        1. Linear (lin)
        2. Log (log)
        3. Exponential (exp)
        4. Linear through zero (lin0)
    """

    import numpy as np
    from scipy import stats

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

    from scipy import polyfit, polyval

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


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import randn

    n = 50
    x = np.logspace(-50, 30, n)
    y = np.logspace(-30, 50, n)

    linX, linY = calculateRegression(x, y, 'lin')
    #linX2, linY2 = calculatePolyfit(x, y)
    logX, logY = calculateRegression(x, y, 'log')
    expX, expY = calculateRegression(x, y, 'exp')

    plt.figure()
    plt.loglog(x, y, 'b-s', label = 'data')
    plt.loglog(linX, linY, 'g-x', label = 'linear')
    #plt.loglog(linX2, linY2, 'y-+', label = 'linear2')
    #plt.loglog(logX, logY, 'r', label = 'log')
    #plt.loglog(expX, expY, 'k-+', label = 'exponential')
    plt.legend(loc=2)

    plt.show()





