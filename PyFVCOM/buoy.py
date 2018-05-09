"""
Functions to interrogate and extract buoy data from the buoys.db SQLite3
database.

"""

from __future__ import print_function

from datetime import datetime
from pathlib import Path
from warnings import warn

import numpy as np

try:
    import sqlite3
    use_sqlite = True
except ImportError:
    warn('No sqlite standard library found in this python '
         'installation. Some functions will be disabled.')
    use_sqlite = False


def _split_lines(line, remove_empty=False, remove_trailing=False):
    """
    Quick function to tidy up lines in an ASCII file (split on a given separator (default space)).

    Parameters
    ----------
    line : str
        String to split.
    remove_empty : bool, optional
        Set to True to remove empty columns. Defaults to leaving them in.
    remove_trailing : bool, optional
        Set to True to remove trailing empty columns. Defaults to leaving them in.

    Returns
    -------
    y : list
        The split string.

    """

    delimiters = (';', ',', '\t', ' ')
    delimiter = None
    for d in delimiters:
        if d in line:
            delimiter = d
            break

    # Clear out newlines.
    line = line.strip('\n')

    if remove_trailing:
        line = delimiter.join(line.rstrip(delimiter).split(delimiter))

    y = line.split(delimiter)

    if remove_empty:
        y = [i.strip() for i in line.split(delimiter) if i]

    return y


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

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python '
                           'installation. This function (get_buoy_metadata) '
                           'is unavailable.')

    def _dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    meta_info = [False]
    try:
        with sqlite3.connect(db) as con:
            con.row_factory = _dict_factory
            c = con.cursor()
            meta_info = c.execute('SELECT * from Stations').fetchall()
    except sqlite3.Error as e:
        print('Error %s: {}'.format(e.args[0]))

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
    buoy.get_observed_metadata : extract metadata for a buoy time series.

    Notes
    -----
    Search is case insensitive (b0737327 is equal to B0737327).

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python '
                           'installation. This function (get_buoy_data) is '
                           'unavailable.')

    if noisy:
        print('Getting data for {} from the database...'.format(table), end=' ')

    try:
        with sqlite3.connect(db) as con:
            c = con.cursor()
            # I know, using a string is Bad. But it works and it's only me # working with this.
            data = np.asarray(c.execute('SELECT {} FROM {}'.format(','.join(fields), table)).fetchall()).astype(float)
        if noisy:
            print('done.')
    except sqlite3.Error as e:
        print('Error %s:' % e.args[0])
        data = np.asarray([False])

    return data


class Buoy(object):
    """ Generic class for buoy data (i.e. surface time series). """

    def __init__(self, filename, position=(np.nan, np.nan), station=None, missing_value=None, noisy=False):
        """
        Create a buoy object from the given file name.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.
        position : tuple, list, optional
            Longitude/latitude of the current buoy. If omitted, position is np.nan/np.nan.
        station : str, optional
            The name of the current buoy. If omitted, set to None.
        missing_value : float
            Supply a value which is the missing value for this buoy time series.
        noisy : bool, optional
            If True, verbose output is printed to screen. Defaults to False.

        """

        self._file = Path(filename)

        self._debug = False
        self._noisy = noisy
        self._locations = position
        self._site = station
        self._missing = missing_value
        self._time_header = ['Year', 'Serial', 'Jd', 'Time', 'Time_GMT', 'Date_YYMMDD', 'Time_HHMMSS', 'Date/Time_GMT']
        self.data = None
        self.position = None
        self.time = None

        # Get the metadata read in.
        self._slurp_file()
        self.header, self.header_length, self.header_indices = _read_header(self._lines, self._time_header)

    def _slurp_file(self):
        """
        Read in the contents of the file into self so we don't have to read each file multiple times.

        Provides
        --------
        self._lines : list
            The lines in the file, stripped of newlines and leading/trailing whitespace and split based on a trying a
            few common delimiters.

        """

        extension = self._file.suffix
        # Ignore crappy characters by forcing everything to ASCII.
        with self._file.open('r', encoding='ascii', errors='ignore') as f:
            empty = False
            trailing = False
            if extension == '.csv':
                # Probably CEFAS data which has empty columns, so we need to leave them in place. However, we need to
                # remove the trailing empty columns as the headers don't account for those.
                trailing = True
            elif extension == '.txt':
                # Probably WCO data, which is usually space separated, so nuke duplicate spaces.
                empty = True
            elif 'L4' in str(self._file):
                # The other WCO data format.
                empty = False
                trailing = False
            elif 'E1' in str(self._file):
                # The other WCO data format.
                empty = False
                trailing = False

            self._lines = f.readlines()
            self._lines = [_split_lines(i, remove_empty=empty, remove_trailing=trailing) for i in self._lines]

            # If we've left empty columns in, replace them with NaNs. This is not elegant.
            if not empty:
                new_lines = []
                for line in self._lines:
                    new_lines.append([np.nan if i == '' else i for i in line])
                self._lines = new_lines

    def load(self):
        """
        Parse the header and extract the data for the loaded file.

        Provides
        --------
        Adds data and time objects, with the variables and time loaded respectively. The time object also has a
        datetime attribute.

        """

        # Add times.
        self.time = self._ReadTime(self._lines)

        if not any(self.time.datetime):
            return

        # Add positions
        self.position = self._ReadPosition(self._locations, self._site)

        # Grab the data.
        self.data = self._ReadData(self._lines)

        # Replace missing values with NaNs.
        if self._missing is not None:
            for name in dir(self.data):
                if not name.startswith('_'):
                    values = getattr(self.data, name)
                    values[values == self._missing] = np.nan
                    setattr(self.data, name, values)

    class _Read(object):
        def __init__(self, lines, noisy=False):
            """
            Initialise parsing the buoy time series data so we can subclass this for the header and data reading.

            Parameters
            ----------
            lines : list
                The data to parse, read in by Buoys._slurp.
            noisy : bool, optional
                If True, verbose output is printed to screen. Defaults to False.

            Provides
            --------
            Attributes in self which are named for each variable found in `lines'. Each attribute contains a single
            time series as a numpy array.

            """

            self._debug = False
            self._noisy = noisy
            self._lines = lines
            self._time_header = ['Year', 'Serial', 'Jd', 'Time', 'Time_GMT', 'Date_YYMMDD', 'Time_HHMMSS', 'Date/Time_GMT']

            self._header, self._header_length, self._header_indices = _read_header(self._lines, self._time_header)
            self._read()

    class _ReadData(_Read):
        """ Read time series data from a given WCO file. This is meant to be called by the Buoy class. """

        def _read(self):
            """
            Parse the data in self._lines for each of the time series.

            Provides
            --------
            Attributes in self which are named for each variable found in `self._lines'. Each attribute contains a single
            time series as a numpy array.

            """

            # We want everything bar the time column names.
            num_lines = len(self._lines) - self._header_length
            num_columns = len(self._header)
            if num_lines > 1:
                for name in self._header:
                    if name not in self._time_header:
                        name_index = self._header_indices[name]
                        data = []
                        for line in self._lines[self._header_length:]:
                            if name_index >= len(line):
                                data.append(np.nan)
                            else:
                                data.append(line[name_index])
                        try:
                            setattr(self, name.strip().replace(' ', '_').replace('(', '').replace(')', ''), np.asarray(data, dtype=float))
                        except ValueError:
                            # Probably strings so just leave as is.
                            setattr(self, name.strip().replace(' ', '_').replace('(', '').replace(')', ''), np.asarray(data))

    class _ReadTime(_Read):
        """ Extract the time from the given WCO file. This is meant to be called by the Buoy class. """

        def _read(self):
            """
            Parse the data in self._lines for each of the time series.

            Provides
            --------
            Attributes in self which are named for each variable found in `self._lines'. Each attribute contains some
            time data. We also create a datetime attribute which has the times as datetime objects.

            """

            # Try everything in self._time_header values.
            self.time_header = []
            num_lines = len(self._lines) - self._header_length
            num_columns = len(self._header)
            if num_lines > 1:
                for name in self._header:
                    if name in self._time_header:
                        self.time_header.append(name)
                        name_index = self._header_indices[name]
                        data = []
                        for line in self._lines[self._header_length:]:
                            data.append(line[name_index])
                        setattr(self, name.strip().replace(' ', '_').replace('(', '').replace(')', ''), np.asarray(data))

            # Now make datetime objects from the time.
            self.datetime = []
            if hasattr(self, 'Year') and hasattr(self, 'Serial') and hasattr(self, 'Time'):
                # First Western Channel Observatory format.
                for year, doy, time in zip(self.Year, self.Serial, self.Time):
                    self.datetime.append(datetime.strptime('{y}{doy} {hm}'.format(y=year, doy=doy, hm=time), '%Y%j %H.%M'))
            elif hasattr(self, 'Year') and hasattr(self, 'Jd') and hasattr(self, 'Time'):
                # Different Western Channel Observatory format.
                for year, doy, time in zip(self.Year, self.Jd, self.Time):
                    try:
                        self.datetime.append(datetime.strptime('{y}{doy} {hm}'.format(y=year, doy=doy, hm=time), '%Y%j %H.%M'))
                    except ValueError:
                        self.datetime.append(np.nan)
            elif hasattr(self, 'Date_YYMMDD') and hasattr(self, 'Time_HHMMSS'):
                # Another different Western Channel Observatory format.
                for date, time in zip(getattr(self, 'Date_YYMMDD'), getattr(self, 'Time_HHMMSS')):
                    if date == 'nan' and time == 'nan':
                        self.datetime.append(np.nan)
                    else:
                        self.datetime.append(datetime.strptime('{date} {time}'.format(date=date, time=time), '%y%m%d %H%M%S'))
            elif hasattr(self, 'Time_GMT'):
                # CEFAS format.
                for date in getattr(self, 'Time_GMT'):
                    self.datetime.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
            elif hasattr(self, 'Date/Time_GMT'):
                # CCO format.
                for date in getattr(self, 'Date/Time_GMT'):
                    self.datetime.append(datetime.strptime(date, '%d-%b-%Y %H:%M:%S'))

    class _ReadPosition:
        """ Add the position for the buoy. """

        def __init__(self, location, site):
            """
            Grab the data for the given buoy site.

            Parameters
            ----------
            location : tuple, list
                The longitude/latitude of the buoy.
            site : str
                The name of the site we're working on.

            Provides
            --------
            lon, lat : float
                The longitude and latitude of the site.

            """

            self.lon, self.lat = location
            self.name = site


def _read_header(lines, header_names):
    """
    Extract the header columns. Accounts for duplicates.

    Parameters
    ----------
    lines : list
        List of the lines from the file.
    header_names : list
        Header time variable names to search for which define the header.

    Returns
    -------
    header : list
        List of the header names.
    header_length : int
        Number of lines in the header.
    header_indices : dict
        Indices of each header name in `header' with the name as the key.

    """
    header_length = 0
    for count, line in enumerate(lines):
        if any(time in line for time in header_names):
            header_length = count
        else:
            break

    header_length += 1

    # Remove annoying characters from header names.
    header = [i.strip().replace(' ', '_').replace('(', '').replace(')', '') for i in lines[header_length - 1]]

    header_indices = {i: header.index(i) for i in header}

    return header, header_length, header_indices
