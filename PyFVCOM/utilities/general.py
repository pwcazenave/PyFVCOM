import itertools
import numpy as np
import warnings
from html.parser import HTMLParser
from re import sub
from sys import stderr
from traceback import print_exc


class PassiveStore(object):
    """
    We ab(use) this class for nesting objects within a class.

    # Add the following decorator to disable a lot of "Unresolved references" warnings in PyCharm.
    @DynamicAttrs

    """
    def __init__(self):
        """ Make an empty object. """
        pass

    def __iter__(self):
        # Iterate over attributes inside this object which don't start with underscores.
        return (a for a in self.__dict__.keys() if not a.startswith('_'))

    def __eq__(self, other):
        # For easy comparison of classes.
        return self.__dict__ == other.__dict__


def fix_range(a, nmin, nmax):
    """
    Given an array of values `a', scale the values within in to the range
    specified by `nmin' and `nmax'.

    In the case where all the values are identical, the values are returned unchanged.

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

    # In the case where all the values are the same, just return the original values.
    if A != B:
        b = (((D - C) * (a - A)) / (B - A)) + C
    else:
        b = a

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

    return np.unravel_index(index, array_shape)


def flatten_list(nested):
    """ Flatten a list of lists. """
    try:
        flattened = list(itertools.chain(*nested))
    except TypeError:
        # Maybe it's already flat and we've just tried iterating over non-iterables. If so, just return what we
        # got given.
        flattened = nested

    return flattened


def split_string(x, separator=' '):
    """
    Quick function to tidy up lines in an ASCII file (split on spaces and strip consecutive spaces).

    Parameters
    ----------
    x : str
        String to split.
    separator : str, optional
        Give a separator to split on. Defaults to space.

    Returns
    -------
    y : list
        The split string.

    """

    return [i.strip() for i in x.split(separator) if i.strip()]


class ObjectFromDict(object):
    """ Convert a dictionary into an object with attributes named from the dictionary keys. """
    def __init__(self, *initial_data, keys=[], **kwargs):
        """
        Convert a dictionary into an object with attributes named from the dictionary keys.

        Parameters
        ----------
        dictionary : dict
            Dictionary to convert to an object with attributes named from the dictionary keys.
        keys : list, optional
            Supply a list of keys which should be extracted from the given dictionary. All others are ignored.
            Defaults to all keys in the dictionary.

        """

        if len(initial_data) > 1:
            raise ValueError('Supply a single dictionary only to convert to an object.')

        for dictionary in initial_data:
            if not keys:
                keys = dictionary.keys()

            for key in keys:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            if key in keys:
                setattr(self, key, kwargs[key])


class _DeHTMLParser(HTMLParser):
    """ Parse HTML so we can more easily extract sensible information from it. """
    def __init__(self):
        HTMLParser.__init__(self)
        self.__text = []

    def handle_data(self, data):
        text = data.strip()
        if len(text) > 0:
            text = sub('[ \t\r\n]+', ' ', text)
            self.__text.append(text + ' ')

    def handle_starttag(self, tag, attrs):
        if tag == 'p':
            self.__text.append('\n\n')
        elif tag == 'br':
            self.__text.append('\n')

    def handle_startendtag(self, tag, attrs):
        if tag == 'br':
            self.__text.append('\n\n')

    def text(self):
        return ''.join(self.__text).strip()


def cleanhtml(text):
    """ Given some HTML text, remove the markup and leave only the text.

    Parameters
    ----------
    text : str
        The HTML to de-markup.

    Returns
    -------
    cleaned : str
        The cleaned text.

    """
    try:
        parser = _DeHTMLParser()
        parser.feed(text)
        parser.close()
        return parser.text()
    except:
        print_exc(file=stderr)
        return text


def cart2pol(x, y, degrees=False):
    """
    Apparantly this doesn't exist in numpy already. Originally from SO.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if degrees:
        phi = np.mod(np.rad2deg(phi), 360)
    return(rho, phi)


def pol2cart(rho, phi, degrees=False):
    """
    As above.
    """
    if degrees:
        phi = np.deg2rad(phi)

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def make_climate(data, dt_list):
    """
    Make an annual climatology of data. Assumes daily data.

    Parameters
    ----------
    data : array
        The data with the first dimension as the time axis
    dt_list : list like
        List of datetime objects for the dates of the data

    Returns
    -------
    ydays : array
        The yeardays in the climatology
    climate : array
        The climatology for each of the year days
        
    """
    yd_list = [this_dt.timetuple().tm_yday for this_dt in dt_list]
    climate = []
    for this_day in np.unique(yd_list):
        climate.append(np.mean(data[yd_list == this_day, ...], axis=0))

    return np.unique(yd_list), np.asarray(climate)

def expand_climate(dt_list):
    """
    Produces the indices and weights to create a timeseries out of a daily (year day) climatology.
    It gives two indices and weights for each so that daily climatologys can be interpolated to higher resolution
    output

    Parameters
    ----------
    dt_list : n list
        List of datetime objects to create the indices and weights for

    Returns
    ------- 
    climate ind : 2 x n array
        The 
    weights : 2 x n array 
        The climatology for each of the year days
        
    """

    yd_list = np.asarray([(this_dt - dt.datetime(this_dt.year,1,1)).total_seconds()/(60*60*24) for this_dt in dt_list])

    climate_ind = np.asarray(np.vstack([np.floor(yd_list), np.ceil(yd_list)]).T, dtype=int)
    weights = np.vstack([1 - (yd_list - climate_ind[:,0]), yd_list - climate_ind[:,0]]).T

    return climate_ind, weights




def _warn(*args, **kwargs):
    """ Custom warning function which doesn't print the code to screen. """
    # Mainly taken inspiration from https://stackoverflow.com/questions/2187269.
    msg = warnings.WarningMessage(*args, **kwargs)
    print(f'{msg.message} ({msg.filename}:{msg.lineno})')


# Update the warnings module with the custom warning function and then make warn an object in this module.
warnings.showwarning = _warn
warn = warnings.warn

