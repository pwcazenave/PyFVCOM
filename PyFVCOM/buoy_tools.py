"""
Functions to interrogate and extract buoy data from the buoys.db SQLite3
database.

"""

from __future__ import print_function

import inspect

import numpy as np
import sqlite3

from warnings import warn


def get_buoy_metadata(db):
    """
    Extracts the meta data from the buoy database.

    Parameters
    ----------
    db : str
        Full path to the buoy data SQLite database.

    Returns
    -------
    meta_info : list
        List of dicts with keys based on the field names from the Stations
        table. Returns [False] if there is an error.

    """

    def _dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    try:
        con = sqlite3.connect(db)

        con.row_factory = _dict_factory

        c = con.cursor()

        out = c.execute('SELECT * from Stations')

        meta_info = out.fetchall()

    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error %s:' % e.args[0])
            meta_info = [False]

    finally:
        if con:
            con.close()

    return meta_info


def get_buoy_data(db, table, fields, noisy=False):
    """
    Extract the buoy from the SQLite database for a given site.  Specify the
    database (db), the table name (table) of the station of interest.

    Parameters
    ----------
    db : str
        Full path to the buoy data SQLite database.
    table : str
        Name of the table to be extracted (e.g. 'hastings_wavenet_site').
    fields : list
        List of names of fields to extract for the given table, such as
        ['Depth', 'Temperature']. Where no data exists, a column of NaNs will
        be returned (actually Nones, but numpy does the conversion for you).
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    data : ndarray
        Array of the fields requested from the table specified.

    See Also
    --------
    buoy_tools.get_observed_metadata : extract metadata for a buoy time series.

    Notes
    -----
    Search is case insensitive (b0737327 is equal to B0737327).

    """

    if noisy:
        print('Getting data for {} from the database...'.format(table),
              end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            # I know, using a string is Bad. But it works and it's only me
            # working with this.
            c.execute('SELECT {} FROM {}'.format(','.join(fields), table))

            # Now get the data in a format we might actually want to use
            data = np.asarray(c.fetchall())

        if noisy:
            print('done.')

    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error %s:' % e.args[0])
            data = np.asarray([False])

    finally:
        if con:
            con.close()

    return data.astype(float)


# Add for backwards compatibility.
def getBuoyMetadata(*args, **kwargs):
    warn('{} is deprecated. Use get_buoy_metadata instead.'.format(inspect.stack()[0][3]))
    return get_buoy_metadata(*args, **kwargs)


def getBuoyData(*args, **kwargs):
    warn('{} is deprecated. Use get_buoy_data instead.'.format(inspect.stack()[0][3]))
    return get_buoy_data(*args, **kwargs)
