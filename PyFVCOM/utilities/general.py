import numpy as np


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


def ind2sub(array_shape, index):
    """
    NOTE: Just use numpy.unravel_index!

    Replicate the MATLAB ind2sub function to return the subscript values (row,
    column) of the index for a matrix shaped `array_shape'.

    Parameters
    ----------
    array_shape : list, tuple, ndarray
        Shape of the array for which to calculate the indices.
    index : int
        Index in the flattened array.

    Returns
    -------
    row, column : int
        Indices of the row and column, respectively, in the array of shape
        `array_shape'.

    """

    # print('WARNING: Just use numpy.unravel_index!')
    # rows = int(np.array(index, dtype=int) / array_shape[1])
    # # Or numpy.mod(ind.astype('int'), array_shape[1])
    # cols = int(np.array(index, dtype=int) % array_shape[1])
    #
    # return (rows, cols)

    return np.unravel_index(index, array_shape)



