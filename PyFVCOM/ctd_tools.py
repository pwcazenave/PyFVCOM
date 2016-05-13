"""
Functions to interrogate and extract CTD data from the ctd.db SQLite3 database.

"""

from __future__ import print_function

import inspect
import sqlite3

import numpy as np

from warnings import warn


def get_CTD_metadata(db):
    """
    Extracts the meta data from the CTD database.

    Parameters
    ----------
    db : str
        Full path to the CTD data SQLite database.

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
            print('Error {}:'.format(e.args[0]))
            meta_info = [False]

    finally:
        if con:
            con.close()

    return meta_info


def get_CTD_data(db, table, fields, noisy=False):
    """
    Extract the CTD from the SQLite database for a given site. Specify the
    database (db), the table name (table) of the station of interest.

    Parameters
    ----------
    db : str
        Full path to the CTD data SQLite database.
    table : str
        Name of the table to be extracted (e.g. 'b0000001').
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
    ctd_tools.get_observed_metadata : extract metadata for a CTD cast.

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
            print('Error {}:'.format(e.args[0]))
            data = np.asarray([False])

    finally:
        if con:
            con.close()

    return data


def get_ferrybox_data(db, fields, table='PrideOfBilbao', noisy=False):
    """
    Extract the Ferry Box data from the SQLite database for a given route
    (defaults to PrideOfBilbao). Specify the database (db), the table name
    (table) of the station of interest.

    Parameters
    ----------
    db : str
        Full path to the CTD data SQLite database.
    fields : list
        List of names of fields to extract for the given table, such as
        ['salinity', 'temp']. Where no data exists, a column of NaNs will
        be returned (actually Nones, but numpy does the conversion for you).
    table : str, optional
        Name of the table to be extracted (defaults to 'PrideOfBilbao').
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    data : ndarray
        Array of the fields requested from the table specified.

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
            print('Error {}:'.format(e.args[0]))
            data = np.asarray([False])

    finally:
        if con:
            con.close()

    return data


# Add for backwards compatibility.
def getCTDMetadata(*args, **kwargs):
    warn('{} is deprecated. Use get_CTD_metadata instead.'.format(inspect.stack()[0][3]))
    return get_CTD_metadata(*args, **kwargs)


def getCTDData(*args, **kwargs):
    warn('{} is deprecated. Use get_CTD_data instead.'.format(inspect.stack()[0][3]))
    return get_CTD_data(*args, **kwargs)


def getFerryBoxData(*args, **kwargs):
    warn('{} is deprecated. Use get_ferrybox_data instead.'.format(inspect.stack()[0][3]))
    return get_ferrybox_data(*args, **kwargs)
