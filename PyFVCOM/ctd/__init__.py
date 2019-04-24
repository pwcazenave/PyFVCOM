"""
Functions to interrogate and extract CTD data from the ctd.db SQLite3 database.

"""

from __future__ import print_function

from datetime import datetime
from pathlib import Path
from collections import OrderedDict

import numpy as np
from netCDF4 import Dataset

from PyFVCOM.utilities.general import split_string, ObjectFromDict, cleanhtml, flatten_list, warn

use_sqlite = True
try:
    import sqlite3
except ImportError:
    warn('No sqlite standard library found in this python installation. Some functions will be disabled.')
    use_sqlite = False


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

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_CTD_metadata)'
                           ' is unavailable.')

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
    ctd.get_observed_metadata : extract metadata for a CTD cast.

    Notes
    -----
    Search is case insensitive (b0737327 is equal to B0737327).

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_CTD_data)'
                           ' is unavailable.')

    if noisy:
        print('Getting data for {} from the database...'.format(table), end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            # I know, using a string is Bad. But it works and it's only me working with this.
            c.execute('SELECT {} FROM {}'.format(','.join(fields), table))

            # Now get the data in a format we might actually want to use
            data = np.asarray(c.fetchall()).astype(float)

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


def _split_wco_lines(line):
    """
    Custom function based on invocations of split_string to handle the inconsistently formatted WCO data. It
    currently tries to split based on (in this order): ';', ',', ' '.

    Parameters
    ----------
    line : str
        Line the split

    Returns
    -------
    line_list : list
        The split line.

    """

    delimiters = (';', ',', ' ')
    delimiter = None
    for d in delimiters:
        if d in line:
            delimiter = d
            break

    return split_string(line, delimiter)


class CTD(object):
    """ Generic class for CTD data. """

    def __init__(self, filename, noisy=False):
        """
        Initialise a CTD object.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.
        noisy : bool, optional
            If True, verbose output is printed to screen. Defaults to False.

        """

        self._debug = False
        self._noisy = noisy
        self.data = None  # assume we're not reading data unless self.read() is called.

        self._file = Path(filename)
        # Read the header into self so we can pass it around more easily. Really, having the header as a dictionary
        # is probably the most sensible thing.
        self.header = self._ParseHeader(self._file, self._noisy).header
        # Store the variable names in here for ease of access.
        self.variables = ObjectFromDict(self.header, keys=['units', 'names', 'long_name'])
        # These two functions extract bits of information from the header we've just parsed.
        self.time = ObjectFromDict(self.header, keys=['datetime', 'time_units', 'interval'])
        self.position = ObjectFromDict(self.header, keys=['lon', 'lat', 'depth', 'sensor'])

    def load(self):
        """
        Generic wrapper around the two main BODC file formats I'm interested in (QXF. vs LST, essentially netCDF vs.
        ASCII).

        Provides
        --------
        Object with the data, names, units and long_name attributes.

        """

        self.data = self._ReadData(self.header)

        # Update the time object in case we've read time information from the data columns.
        self.time = ObjectFromDict(self.header, keys=['datetime', 'time_units', 'interval'])

    def write(self, filename):
        """
        Wrapper around the two main BODC file formats I'm interested in (QXF. vs LST, essentially netCDF vs.
        ASCII). This tries to guess which format you want based on file extension, and as such, is as flaky.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to which to write data.

        """
        try:
            suffix = Path(filename).suffix
        except ValueError:
            suffix = filename.suffix

        if suffix == '.qxf':
            self._write_qxf(filename)
        elif suffix == '.lst':
            self._write_lst(filename)
        else:
            raise ValueError('Unsupported file extension: {}, supply either .lst or .qxf'.format(suffix))

    def _write_lst(self, filename):
        """
        Write CTD data to an LST-formatted file.

        If we've read in mulitple CTD casts (i.e. the various self.data attributes are lists of arrays), then we write
        one file per cast with a suitably zero-padded number appended for each cast.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to which to write out.

        """

        # Parse the data for relevant metadata.
        filename = Path(filename)
        if isinstance(self.variables.names[0], list):
            # Append a number per file we create. This is for the case where we've read in WCO data.
            parent = filename.parent
            stem = filename.stem
            suffix = filename.suffix
            number_of_casts = len(self.header['record_indices'])
            precision = len(str(number_of_casts))
            file_string = '{st}_{ct:0{pr}d}{sx}'
            file_names = [Path(parent) / Path(file_string.format(st=stem, ct=i + 1, pr=precision, sx=suffix)) for i in range(number_of_casts)]
        else:
            file_names = [filename]

        for counter, current_file in enumerate(file_names):
            # We need to account for the crappy date/time columns in the WCO data (which we haven't read in to each
            # self.data attribute).
            num_fields = len(self.variables.names[counter])
            if 'mm_dd_yyyy' in self.variables.names[counter] and 'hh_mm_ss' in self.variables.names[counter]:
                num_fields -= 2
            num_samples = len(getattr(self.data, self.variables.names[counter][0])[counter])

            # For the Header field, there seems to be 12 lines of standard headers plus pairs of lines for each
            # variable.
            bodc_header_lines = 12 + (num_fields * 2)

            if self.time.datetime[counter] is not None:
                start = self.time.datetime[counter].strftime('%Y%m%d%H%M%S')
            else:
                start = self.time.datetime[counter]
            if self.position.lon[counter] is not None:
                lon = np.mean(self.position.lon[counter])
            else:
                lon = self.position.lon[counter]
            if self.position.lat[counter] is not None:
                lat = np.mean(self.position.lat[counter])
            else:
                lat = self.position.lat[counter]
            northsouth = 'N'
            westeast = 'E'
            if lat is not None and lat < 0:
                northsouth = 'S'
            if lon is not None and lon < 0:
                westeast = 'W'

            # Some stuff, we'll just ignore/make up for now.
            try:
                sensor = self.position.sensor[counter]
                # If this sensor set is a pair of Nones, then use the depth as below.
                if sensor == [None, None]:
                    sensor = [np.nanmin(self.position.depth[counter]), np.nanmax(self.position.depth[counter])]
            except AttributeError:
                sensor = [np.nanmin(self.position.depth[counter]), np.nanmax(self.position.depth[counter])]

            # The BODC format is... unique. Try to replicate it to the extent which is useful for me. May not be 100%
            # compatible with actual BODC formatting.
            with current_file.open('w') as f:
                # Naturally, we start with three blank lines.
                f.write('\n\n\n')
                header_string = 'BODC Request Format Std. V1.0           Headers=  {} Data Cycles=   {}\n'
                f.write(header_string.format(bodc_header_lines, num_samples))
                series_string = 'Series=      {}                     Produced: {}\n'
                f.write(series_string.format(current_file.stem, datetime.now().strftime('%d-%b-%Y')))  # dummy data for now
                f.write('Id                       {}\n'.format(current_file.stem))  # more dummy data
                position_string = '{deglat:03d}d{minlat:.1f}m{hemilat}{deglon:03d}d{minlon:.1f}m{hemilon}'
                position = position_string.format(deglat=abs(int(lat)),
                                                  minlat=(abs(lat) - abs(int(lat))) * 60,
                                                  hemilat=northsouth,
                                                  deglon=abs(int(lon)),
                                                  minlon=(abs(lon) - abs(int(lon))) * 60,
                                                  hemilon=westeast)
                f.write('{position}                     start:{start}\n'.format(position=position, start=start))
                format_string = 'Dep: floor {depth:.1f} sensor    {sensor_1:.1f}  {sensor_2:.1f} ' \
                                'Nom. sample int.:    {interval:.1f} {units}\n'
                f.write(format_string.format(depth=self.position.depth[counter],
                                             sensor_1=sensor[0],
                                             sensor_2=sensor[1],
                                             interval=self.time.interval[counter],
                                             units=self.time.time_units[counter]))
                f.write('    {} Parameters included:\n'.format(num_fields))
                # Now we're into the headers for each variable.
                f.write('Parameter f    P    Q Absent Data Value Minimum Value  Maximum Value       Units\n')
                for name in self.variables.names[counter]:
                    # Skip WCO time data columns as we haven't saved those with _ReadData._read_wco().
                    if name in ('mm_dd_yyyy', 'hh_mm_ss'):
                        continue
                    header_string = '{name:8s}  {f} {P} {Q} {missing:.2f} {min:.2f} {max:.2f} {unit}\n'
                    f.write(header_string.format(name=name[:8],
                                                 f='Y',
                                                 P=0,
                                                 Q=0,
                                                 missing=-1,
                                                 min=np.nanmin(getattr(self.data, name)[counter]),
                                                 max=np.nanmax(getattr(self.data, name)[counter]),
                                                 unit=self.variables.units[counter][name]))
                    f.write('{}\n'.format(self.variables.long_name[counter][name][:80]))  # maximum line length is 80 characters
                f.write('\n')
                f.write('Format Record\n')
                f.write('(some nonsense for now)\n')
                # A few more new lines in case we're missing some.
                f.write('\n\n\n')
                # The data header.
                f.write('  Cycle     {}\n'.format('   '.join([i[:8] for i in self.variables.names[counter] if i not in ('mm_dd_yyyy', 'hh_mm_ss')])))
                f.write('Number             {}\n'.format('          '.join('f' * num_fields)))
                # Now add the data.
                cycle = ['{})'.format(i) for i in np.arange(num_samples) + 1]
                data = []
                for name in self.variables.names[counter]:
                    # Skip WCO time data columns as we haven't saved those with _ReadData._read_wco().
                    if name in ('mm_dd_yyyy', 'hh_mm_ss'):
                        continue
                    data.append(['{}'.format(i) for i in getattr(self.data, name)[counter]])
                if self._debug:
                    if np.diff([len(i) for i in data]).max() == 1:
                        raise ValueError('broken data')
                data = np.column_stack((cycle, np.asarray(data).T))

            # Must be possible to dump a whole numpy array in one shot...
            with current_file.open('ab') as f:
                np.savetxt(f, data, fmt=['%10s'] * (num_fields + 1))

    def _write_qxf(self, filename):
        """
        Write CTD data to a QXF (netCDF) file.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to which to write out.

        """
        pass

    # Hereafter all the heavy lifting.
    class _ParseHeader(object):
        """ Parse a file for the information found in the header. """

        def __init__(self, filename, noisy=False):
            """
            Based on file extension, call one of the header parsers below. For now, that's only BODC LST and QXF
            (netCDF4) formats as well as the WCO time series data.

            Parameters
            ----------
            filename : str, pathlib.Path
                The file name to read in.
            noisy : bool, optional
                If True, verbose output is printed to screen. Defaults to False.

            Returns
            -------
            header : dict
                Dictionary of the header information in the BODC LST-formatted file. This includes a standard set of keys
                plus those defined in the header itself. The standard keys are:
                    'file_name' - the file name from which we've read.
                    'names' - the variable names in the file.
                    'units' - the variable units.
                    'long_name' - the variable descriptions.

            """
            self._noisy = noisy
            self._file = Path(filename)
            self.header = {}

            if self._file.suffix == '.lst':
                self._read_lst_header()
            elif self._file.suffix == '.qxf':
                self._read_qxf_header()
            elif self._file.suffix == '.txt':
                # I don't like this one bit.
                self._read_wco_header()
            else:
                # Add more readers here.
                pass

        def _read_lst_header(self):
            """
            Get the BODC LST-formatted header. Store the information as a dictionary to make extracting relevant
            information easier.

            Provides
            --------
            header : dict
                Dictionary of the header information in the BODC LST-formatted file. This includes a standard set of
                keys plus those defined in the header itself. The standard keys are:
                    'file_name' - the file name from which we've read.
                    'names' - the variable names in the file.
                    'units' - the variable units.
                    'long_name' - the variable descriptions.

            """

            self.header['file_name'] = str(self._file)  # keep a record of the file we're opening.
            self.header['names'] = []  # store the variable names
            self.header['units'] = {}  # the variables' units
            self.header['long_name'] = {}  # the variable descriptions

            with self._file.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        line_list = split_string(line)
                        if line.startswith('BODC'):
                            self.header['header_length'] = int(line_list[6])
                            self.header['num_records'] = int(line_list[9])
                        elif line.startswith('Series'):
                            self.header['series_id'] = line_list[1]
                        elif 'start:' in line:
                            # Extract degrees, minutes, hemisphere for latitude and longitude. The least awful way of
                            # dealing with the total mess that is this formatting is to convert the whole string
                            # into a list of characters and read through them one by one, pulling out numbers where
                            # necessary otherwise grabbing the letters.
                            raw_position = list(line_list[0])
                            numbers = [i if i.isnumeric() or i == '.' else ' ' for i in raw_position]
                            numbers = [float(i) for i in ''.join(numbers).strip().split(' ') if i]
                            letters = [i if i.isalpha() else '' for i in raw_position]
                            letters = [i for i in ''.join(letters) if i]
                            nice_position = [numbers[0], numbers[1], letters[2], numbers[2], numbers[3], letters[-1]]
                            # More useful for us are positions as decimal degrees.
                            self.header['lon'] = nice_position[3] + (nice_position[4] / 60)
                            if nice_position[5] == 'W':
                                self.header['lon'] = -self.header['lon']
                            self.header['lat'] = nice_position[0] + (nice_position[1] / 60)
                            if nice_position[2] == 'S':
                                self.header['lat'] = -self.header['lat']

                            start = line_list[1].split(':')[1]
                            self.header['datetime'] = [datetime.strptime(start, '%Y%m%d%H%M%S')]
                        elif line.startswith('Dep'):
                            if 'floor' in line:
                                # We're missing the depth information, so set it to NaN here.
                                if line_list[2] == 'sensor':
                                    self.header['depth'] = np.nan
                                    self.header['sensor'] = np.nan
                                else:
                                    self.header['depth'] = float(line_list[2])
                                    self.header['sensor'] = [float(i) for i in line_list[4:6]]
                            else:
                                self.header['depth'] = float(line_list[1])
                            interval_to_check = line_list[-2].split(':')[-1]
                            if interval_to_check == 'no':
                                self.header['interval'] = '-'
                            else:
                                self.header['interval'] = float(interval_to_check)
                            self.header['time_units'] = line_list[-1]
                        elif 'included' in line:
                            # Get the number of parameters in this file.
                            self.header['num_fields'] = int(line_list[0])
                            # Read one more line here and then break out here as we handle reading the metadata in a
                            # separate loop.
                            next(f)
                            break

                # Extract the information about each variable.
                for record in range(self.header['num_fields']):
                    line = next(f).strip()
                    if line:
                        line_list = split_string(line)
                        self.header['names'].append(line_list[0])
                        self.header['units'][self.header['names'][-1]] = line_list[-1]
                        # Skip to the next line now so we can get the long_name value.
                        line = next(f).strip()
                        self.header['long_name'][self.header['names'][-1]] = line

        def _read_qxf_header(self):
            """
            Get the BODC QXF-formatted (netCDF) header. Store the information as a dictionary to make extracting
            relevant information easier.

            Provides
            --------
            header : dict
                Dictionary of the header information in the BODC LST-formatted file. This includes a standard set of
                keys plus those defined in the header itself. The standard keys are:
                    'file_name' - the file name from which we've read.
                    'names' - the variable names in the file.
                    'units' - the variable units.
                    'long_name' - the variable descriptions.

            """

            self.header['file_name'] = str(self._file)  # keep a record of the file we're opening.
            self.header['names'] = []  # store the variable names
            self.header['units'] = {}  # the variables' units
            self.header['long_name'] = {}  # the variable descriptions
            self.header['lon'] = []
            self.header['lat'] = []
            self.header['datetime'] = []
            self.header['time_units'] = []
            self.header['interval'] = []
            self.header['depth'] = []
            self.header['sensor'] = []

            # As ever, nothing's easy with the BODC data. The QXF files are actually just netCDF files, and yet,
            # there is almost no useful information in them (no names, units, anything). However, what BODC do do is
            # instead supply a HTML file which has this information in it (I wish I was making this up). So,
            # what we'll do is silently open that file (if it exists) and try to grab as much useful information as
            # possible from it. If it's not there, well, we'll just put blank information everywhere.

            # Drop the b prefix and leading zeros for the HTML file name.
            html_info = Path(self._file.parent, '{}.html'.format(self._file.stem[1:].lstrip('0')))
            with Dataset(self.header['file_name'], 'r') as ds:
                self.header['names'] = list(ds.variables)
                self.header['num_fields'] = len(self.header['names'])
                # Populate everything with Nones or whatever we can gather from the netCDF. This may be overwritten
                # by the HTML file scraping.
                for var in self.header['names']:
                    self.header['num_records'] = len(np.ravel(ds.variables[var][:]))
                    self.header['units'][var] = None
                    self.header['long_name'][var] = None
                    self.header['lon'].append(None)
                    self.header['lat'].append(None)
                    self.header['depth'].append(None)
                    self.header['datetime'].append([None, None])
                    self.header['time_units'].append(None)
                    self.header['interval'].append(None)
                    self.header['sensor'].append(None)

                if html_info.exists():
                    # Ignore crappy characters by forcing everything to ASCII.
                    with html_info.open('r', encoding='ascii', errors='ignore') as html:
                        try:
                            lines = html.readlines()
                        except UnicodeDecodeError as e:
                            # Something weird in the file. Skip out.
                            if self._noisy:
                                print('Corrupt metadata file {}'.format(str(html_info)))
                            return
                        cleanlines = [cleanhtml(line) for line in lines]
                        cleanlines = [i for i in cleanlines if i]
                        # Iterate through the cleaned HTML and extract information from certain keywords.
                        keywords = ('Longitude', 'Latitude',
                                    'Start Time (yyyy-mm-dd hh:mm)', 'End Time (yyyy-mm-dd hh:mm)',
                                    'Nominal Cycle Interval', 'Minimum Sensor Depth', 'Maximum Sensor Depth',
                                    'Sea Floor Depth', 'BODC Series Reference')
                        mapped_names = ('lon', 'lat',
                                        'datetime1', 'datetime2',
                                        'interval', 'sensor1', 'sensor2',
                                        'depth', 'series_id')
                        missed = 0
                        for mapped, key in zip(mapped_names, keywords):
                            try:
                                self.header[mapped] = cleanlines[cleanlines.index(key) + 1]
                            except ValueError:
                                # Missing this piece of information. Keep trying the others, but keep a track of how
                                # many we've missed. If that number equals the length of those we were trying to get,
                                # then bail out as there's obviously something wrong with the HTML file (truncated,
                                # nonsense etc.).
                                missed += 1

                        # Clean the positions a bit.
                        try:
                            lon, lon_hemisphere = self.header['lon'].split(' ')[:2]
                            lat, lat_hemisphere = self.header['lat'].split(' ')[:2]
                            lon = float(lon.strip())
                            lat = float(lat.strip())
                            if lon_hemisphere == 'W':
                                lon = -lon
                            if lat_hemisphere == 'S':
                                lat = -lat
                            self.header['lon'] = lon
                            self.header['lat'] = lat
                        except AttributeError:
                            # This is probably a list of Nones, so just skip it. This happens when we've got a
                            # corrupted HTML metadata file.
                            pass

                        # We haven't found anything useful, so just quit now.
                        if missed == len(keywords):
                            if self._noisy:
                                print('Insufficient useful metadata in file {}'.format(str(html_info)))
                            return

                        # sensor1 and sensor2 need to be merged into sensor. Likewise, datetime1 and datetime2 need
                        # to be made into a single list.
                        self.header['sensor'] = [self.header['sensor1'], self.header['sensor2']]
                        self.header['datetime'] = [self.header['datetime1'], self.header['datetime2']]
                        # Check we've got a valid end time. If not, remove it from the list.
                        if self.header['datetime2'] != '-':
                            # Assume the format is '%Y-%m-%d %H:%M' since that's what we've search with above.
                            self.header['datetime'] = [datetime.strptime(i, '%Y-%m-%d %H:%M') for i in self.header['datetime']]
                        else:
                            self.header['datetime'] = [datetime.strptime(self.header['datetime1'], '%Y-%m-%d %H:%M')]
                        # Remove the temporary keys from the header.
                        for var in ('sensor1', 'sensor2', 'datetime1', 'datetime2'):
                            self.header.pop(var, None)

                        # Get some useful information about the variables.
                        description_indices = [cleanlines.index('Parameters'), cleanlines.index('Definition of Rank')]
                        variable_info = cleanlines[description_indices[0] + 1:description_indices[1]]
                        # To get the number of columns in the table, revert to the raw HTML so we can see the tags
                        # indicated then end of the header row.
                        header_start = [c for c, i in enumerate(lines) if 'BODC CODE' in i][0]  # use the first hit
                        num_columns = [c for c, i in enumerate(lines[header_start:]) if i.strip() == '</tr>'][0]
                        header_names = variable_info[:num_columns]
                        variable_info = variable_info[num_columns:]
                        variables = {i: None for i in header_names}
                        for header_index, name in enumerate(header_names):
                            variables[name] = variable_info[header_index::len(header_names)]
                        # Now convert those into my format.
                        if 'Units' in variables:
                            self.header['units'] = variables['Units']
                        if 'Title' in variables:
                            self.header['long_name'] = variables['Title']
                        if 'BODC CODE' in variables:
                            self.header['names'] = variables['BODC CODE']

        def _read_wco_header(self):
            """
            Get the Western Channel Observatory CTD header. Store the information as a dictionary to make extracting
            relevant information easier.

            Provides
            --------
            header : dict
                Dictionary of the header information in the BODC LST-formatted file. This includes a standard set of
                keys plus those defined in the header itself. The standard keys are:
                    'file_name' - the file name from which we've read.
                    'names' - the variable names in the file.
                    'units' - the variable units.
                    'long_name' - the variable descriptions.

            """

            # Hardcoded WCO buoy positions for those files which are missing those data.
            wco_positions = {'L4': {'lon': -4.21495, 'lat': 50.25015}, 'E1': {'lon': -4.365, 'lat': 50.0355}}

            self.header['file_name'] = str(self._file)  # keep a record of the file we're opening.
            self.header['names'] = []  # store the variable names
            self.header['units'] = []  # list of the variables' units
            self.header['long_name'] = []  # the variable descriptions (list of dictionaries)
            self.header['depth'] = []
            self.header['sensor'] = []  # again, mostly Nones for the WCO data
            self.header['record_indices'] = []  # at what line numbers are the CTD headers?
            self.header['num_records'] = []  # how many samples per cast?
            self.header['num_fields'] = []  # how many variables per cast?
            self.header['lon'] = []
            self.header['lat'] = []
            self.header['datetime'] = []
            self.header['time_units'] = []  # ignored here
            self.header['interval'] = []  # ignored here
            self.header['units'] = []  # list of dictionaries
            self.header['series_id'] = []  # make up a series ID based on file name.

            # Given the state of the WCO data, I need to hard code some useful information about the variables (long
            # names, units etc.)
            units = {'DepSM': 'metres',
                     'Tv290C': 'unknown',
                     'CStarTr0': 'unknown',
                     'Par': 'unknown',
                     'V3': 'unknown',
                     'Latitude': 'degrees north',
                     'Longitude': 'degrees east',
                     'Density00': 'kg/m3',
                     'Sbeox0Mm_Kg': 'unknown',
                     'Sbox0Mm_Kg': 'unknown',
                     'Sal00': 'PSU',
                     'Nbin': '-',
                     'Flag': '-',
                     'CStarTr0': 'unknown',
                     'TurbWETntu0': 'unknown',
                     'FlECO-AFL': 'unknown'}
            long_name = {'DepSM': 'Depth below surface',
                         'Tv290C': 'unknown',
                         'CStarTr0': 'unknown',
                         'Par': 'Photosynthetically Active Radiation',
                         'V3': 'unknown',
                         'Latitude': 'Latitude',
                         'Longitude': 'Longitude',
                         'Density00': 'Water density',
                         'Sbeox0Mm_Kg': 'unknown',
                         'Sbox0Mm_Kg': 'unknown',
                         'Sal00': 'Water salinity',
                         'Nbin': 'unknown',
                         'Flag': 'unknown',
                         'CStarTr0': 'unknown',
                         'TurbWETntu0': 'unknown',
                         'FlECO-AFL': 'unknown'}

            ctd_counter = -1
            with self._file.open('r') as f:
                for line in f:
                    ctd_counter += 1
                    line = line.strip()
                    if line:
                        line_list = _split_wco_lines(line)
                        if 'DepSM' in line_list:
                            # We've got a new CTD cast.
                            self.header['num_fields'].append(len(line_list))
                            self.header['record_indices'].append(ctd_counter)
                            # Remove illegal characters (specifically / and :) from variable names.
                            line_list = [i.replace('/', '_').replace(':', '_') for i in line_list]
                            self.header['names'].append(line_list)

                            # Get the first line of data so we can check the header is in order. Once we've got that,
                            # we can begin extracting data. Save the header list so we can find indices for data.
                            header_list = line_list
                            line = next(f)
                            line_list = _split_wco_lines(line)

                            # First, check the number of columns in the headers matches the number of columns in the
                            # data. Yes, that's right folks, sometimes the header doesn't have all the variables
                            # listed!
                            if len(line_list) != len(self.header['names'][-1]):
                                # Now, this is a tricky one to solve. My experience is that this is usually because
                                # the date/time headers are missing. So, we'll search for things formatted dd/mm/yyyy
                                # and hh:mm:ss and insert the headers accordingly.
                                new_indices = []
                                new_headers = []
                                for position, entry in enumerate(line_list):
                                    if len(entry.split('/')) == 3:
                                        new_indices.append(position)
                                        new_headers.append('mm_dd_yyyy')
                                    if len(entry.split(':')) == 3:
                                        new_indices.append(position)
                                        new_headers.append('hh_mm_ss')
                                for header, pos in zip(new_headers, new_indices):
                                    self.header['names'][-1].insert(pos, header)

                                # This might also be because we have the spurious zeros in the second column. Add a
                                # new header which means we can process this later.
                                if line_list[1] == '0.00':
                                    self.header['names'][-1].insert(1, 'Zeros')

                            # In order to make the header vaguely usable, grab the initial time and position for this
                            # cast.
                            lon_idx, lat_idx, date_idx, time_idx = None, None, None, None
                            if 'Longitude' in header_list:
                                lon_idx = header_list.index('Longitude')
                            if 'Latitude' in header_list:
                                lat_idx = header_list.index('Latitude')
                            if 'mm_dd_yyyy' in header_list:
                                date_idx = header_list.index('mm_dd_yyyy')
                            if 'hh_mm_ss' in header_list:
                                time_idx = header_list.index('hh_mm_ss')

                            if lon_idx is None and lat_idx is None:
                                # Assume some file format for the data and use that for the position.
                                station = self._file.stem.split('_')[-1]
                                try:
                                    self.header['lon'].append(wco_positions[station]['lon'])
                                    self.header['lat'].append(wco_positions[station]['lat'])
                                except KeyError:
                                    self.header['lon'].append(np.nan)
                                    self.header['lat'].append(np.nan)
                            else:
                                self.header['lon'].append(float(line_list[lon_idx]))
                                self.header['lat'].append(float(line_list[lat_idx]))

                            if date_idx is None:
                                self.header['datetime'].append(None)
                            else:
                                datetime_string = ' '.join((line_list[date_idx], line_list[time_idx]))
                                self.header['datetime'].append(datetime.strptime(datetime_string, '%m/%d/%Y %H:%M:%S'))
                            if time_idx is None:
                                self.header['time_units'].append(None)
                            else:
                                self.header['time_units'].append('datetime')
                            # Some sampling interval. Set to zero as the casts are relatively quick (a few minutes,
                            # tops).
                            self.header['interval'].append(0)
                            self.header['units'].append({})
                            for unit in self.header['names'][-1]:
                                if unit in units:
                                    self.header['units'][-1].update({unit: units[unit]})
                                else:
                                    self.header['units'][-1].update({unit: 'unknown'})
                            self.header['long_name'].append({})
                            for l_name in self.header['names'][-1]:
                                if l_name in long_name:
                                    self.header['long_name'][-1].update({l_name: units[l_name]})
                                else:
                                    self.header['long_name'][-1].update({l_name: '-'})

                            # The WCO data have lots of sensors, so just put some placeholders in for the time being.
                            self.header['sensor'].append([None, None])

                            # Manually increment the counter here as we're skipping a line and it would otherwise be
                            # off by one.
                            ctd_counter += 1

                            # Some casts have the same data repeated in the files, because why not! So, remove those
                            # duplicate names and update header information accordingly.
                            self.header['names'][-1] = list(OrderedDict.fromkeys(self.header['names'][-1]))
                            self.header['num_fields'][-1] = len(self.header['names'][-1])

            # Make a set of series IDs for each cast. Pad with an appropriate number of zeros.
            prefix = Path(self.header['file_name']).stem
            number = len(self.header['names'])
            precision = len(str(number))
            self.header['series_id'] = ['{pref}_{id:0{pr}}'.format(pref=prefix, pr=precision, id=i + 1) for i in range(number)]
            # Get the number of records per cast. Offset by one since we count between headers.
            self.header['num_records'] = np.diff(np.concatenate((self.header['record_indices'], [ctd_counter]))) - 1
            # Get the depths for each cast too. Some WCO data store the depth in ascending order, some in descending
            # order. So, grab the first and last of each cast and find the largest value.
            with self._file.open('r') as f:
                lines = f.readlines()
                headers = zip(self.header['names'], self.header['record_indices'], self.header['num_records'])
                for names, cast_index, record_length in headers:
                    # We need two depth values here: the one immediately after the header and the one immediately
                    # before the next header.
                    depth_index = names.index('DepSM')
                    first_line_list = _split_wco_lines(lines[cast_index + 1])
                    last_line_list = _split_wco_lines(lines[cast_index + record_length])
                    try:
                        first_depth = float(first_line_list[depth_index])
                    except ValueError:
                        # The data are probably crappy. Just set the maximum depth to NaN.
                        first_depth = np.nan
                    try:
                        last_depth = float(last_line_list[depth_index])
                    except ValueError:
                        # The data are probably crappy. Just set the maximum depth to NaN.
                        last_depth = np.nan
                    # Also add the last line of this cast's value.
                    # line_list = _split_wco_lines(lines[offset_index + self.header['num_records'][counter]])
                    # depth_index = self.header['names'][-1].index('DepSM')
                    # last_depth = float(line_list[depth_index])
                    self.header['depth'].append(np.nanmax((first_depth, last_depth)))

    class _ReadData(object):

        def __init__(self, header):
            """
            Read data for the file whose metadata is given in `header'. We don't read the data in when we parse the
            header as doing this as discrete steps means we can efficiently filter data which don't match some criteria
            of interest; loading the whole data set would make this a lot slower.

            Parameters
            ----------
            header : dict
                Header parsed with _ParseHeader().

            """

            self._debug = False

            file = Path(header['file_name'])
            suffix = file.suffix
            if suffix == '.lst':
                self._read_lst(header)
            elif suffix == '.qxf':
                self._read_qxf(header)
            elif suffix == '.txt':
                # I don't like this one bit.
                self._read_wco(header)
            else:
                # Add more readers here.
                pass

        def _read_lst(self, header):
            """
            Read the given LST-formatted file and process the data accordingly.

            Parameters
            ----------
            header : dict
                Header parsed with _ParseHeader().

            Provides
            --------
            Each variable is an object in self whose names are based on the header information extracted in
            _ParseHeader().

            """

            file = Path(header['file_name'])

            with file.open('r') as f:
                # In principle, we should be able to use the self.header['header_length'] value to skip the header,
                # but this number doesn't seem to include blank lines, so we're going to ignore anything until we hit
                # the data itself.
                at_data = False
                had_time = False  # do we have time data?
                datetimes = []  # in case we've got time data
                for line in f:
                    line_list = split_string(line.strip())
                    if line.strip():
                        if line_list[0] == 'Cycle':
                            # Next line is the start of the data. Grab the headers so we know where to put what.
                            columns = line_list[1:]  # skip the cycle column
                            at_data = True
                            continue
                        if line_list[0] == 'Number':
                            # The second data header line. We could use this to figure out the formats of the data,
                            # but doesn't seem worth the effort right now. Skip it.
                            continue
                        if at_data:
                            # Some data have a couple of time columns, usually called 'Date' and 'Time'. Convert them to datetime objects.
                            if 'Date' in columns and 'Time' in columns:
                                had_time = True
                                # A quick grep of my LST files shows the date and time formats are %Y/%m/%d and
                                # %H:%M:%S. Offset the indices by one to account for the omitted 'Cycle'.
                                date_index = columns.index('Date') + 1
                                time_index = columns.index('Time') + 1
                                datetimes.append(datetime.strptime(' '.join((line_list[date_index], line_list[time_index])), '%Y/%m/%d %H:%M:%S'))

                            for name_index, name in enumerate(columns):
                                if name in ('Date', 'Time'):
                                    continue
                                # Use the cycle number to get the index for the data. Offset the name_index by 1 to
                                # account for the missing Cycle column.
                                if not hasattr(self, name):
                                    setattr(self, name, np.zeros(header['num_records']))
                                data_index = int(line_list[0].replace(')', '')) - 1
                                getattr(self, name)[data_index] = float(line_list[name_index + 1])

                # If we found date/time data, dump it into the header.
                if had_time:
                    header['datetime'] = datetimes

        def _read_qxf(self, header):
            """
            Read the given QXF-formatted (netCDF) file and process the data accordingly.

            Parameters
            ----------
            header : dict
                Header parsed with _ParseHeader().

            Provides
            --------
            Each variable is an object in self whose names are based on the header information extracted in
            _ParseHeader().

            """
            with Dataset(header['file_name'], 'r') as ds:
                # Grab each variable and dump it into self. If we have more than 1 dimension in the input file,
                # repeat 1D arrays to match the 2D array shapes.
                for variable in ds.variables:
                    setattr(self, variable, ds.variables[variable][:].ravel())  # make everything 1D
                    if len(ds.variables[variable].dimensions) == 1 and len(ds.dimensions) > 1:
                        # In order to make all arrays the same shape, we repeat each 1D one to match the 2D sizes.
                        # We also do this for the Cycle array.
                        num_time = ds.dimensions['primary'].size
                        num_depth = ds.dimensions['secondary'].size
                        # If the number of dimensions in the netCDF is more than 1, then we're assuming the data are
                        # 2D (time, depth) and we need a time-resolved depth array.
                        # Use the appropriate dimension when tiling.
                        if ds.variables[variable].dimensions[0] == 'primary':
                            setattr(self, variable, np.tile(ds.variables[variable][:], num_depth))
                        elif ds.variables[variable].dimensions[0] == 'secondary':
                            setattr(self, variable, np.tile(ds.variables[variable][:], num_time))

        def _read_wco(self, header):
            """
            Read the given Western Channel Observatory-formatted annual file and process the data accordingly.

            Parameters
            ----------
            header : dict
                Header parsed with _ParseHeader().

            Provides
            --------
            Each variable is an object in self whose names are based on the header information extracted in
            _ParseHeader().

            """

            file = Path(header['file_name'])

            # Use the header['record_indices'] and header['num_records'] to read each CTD cast into a list called
            # self.<variable_name>.
            variable_names = np.unique(flatten_list(header['names']))
            # Remove date/time columns.
            variable_names = [i for i in variable_names if i not in ('mm_dd_yyyy', 'hh_mm_ss')]
            for name in variable_names:
                setattr(self, name, [])

            with file.open('r') as f:
                lines = f.readlines()
                for start, length, names in zip(header['record_indices'], header['num_records'], header['names']):
                    data = lines[start + 1:start + length + 1]
                    data = [_split_wco_lines(i) for i in data]
                    # Replace lines with NaNs where we have too few columns.
                    new_data = []
                    for d in data:
                        if len(d) == len(names):
                            new_data.append(d)
                        else:
                            if self._debug:
                                print('Line {} has inconsistently formatted data. Replacing with NaNs.')
                            new_data.append([np.nan] * len(names))
                    data = np.asarray(new_data)

                    if self._debug:
                        print(start, length, start + length + 1)
                        print(data[0, :], data[-1, :])
                    # Dump the data we've got for this cast, excluding the date/time columns.
                    for name in names:
                        if name in ('mm_dd_yyyy', 'hh_mm_ss'):
                            continue
                        if self._debug:
                            print(data[:, names.index(name)].astype(float).shape)
                        getattr(self, name).append(data[:, names.index(name)].astype(float))
                    # Put None in the cumulative list if the current cast is missing a given variable to account for
                    # variables appearing midway through a year.
                    missing_names = set(variable_names) - set(names)
                    for name in missing_names:
                        getattr(self, name).append(None)
