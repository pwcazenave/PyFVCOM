"""
Functions to interrogate and extract CTD data from the ctd.db SQLite3 database.

"""

from __future__ import print_function

from pathlib import Path
from warnings import warn
from datetime import datetime

import numpy as np
from PyFVCOM.utilities.general import split_string, ObjectFromDict

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


class CTD(object):
    """ Generic class for CTD data. """

    def __init__(self, filename):
        """
        Initialise a CTD object.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.

        """

        self._debug = False
        self.data = None  # assume we're not reading data unless self.read() is called.

        self._file = Path(filename)
        # Read the header into self so we can pass it around more easily. Really, having the header as a dictionary
        # is probably the most sensible thing.
        self.header = self._ParseHeader(self._file)
        # Store the variable names in here for ease of access.
        self.variables = ObjectFromDict(self.header.header, keys=['units', 'names', 'long_name'])
        # These two functions extract bits of information from the header we've just parsed.
        self.time = ObjectFromDict(self.header.header, keys=['datetime', 'time_units', 'interval'])
        self.position = ObjectFromDict(self.header.header, keys=['lon', 'lat', 'depth', 'sensor'])

    def load(self):
        """
        Generic wrapper around the two main BODC file formats I'm interested in (QXF. vs LST, essentially netCDF vs.
        ASCII).

        Provides
        --------
        Object with the data, names, units and long_name attributes.

        """

        self.data = self._ReadData(self.header.header)

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
            stem = filename.stem
            suffix = filename.suffix
            number_of_casts = len(self.header.header['record_indices'])
            precision = len(str(number_of_casts))
            file_string = '{st}_{ct:0{pr}d}{sx}'
            file_names = [Path(file_string.format(st=stem, ct=i + 1, pr=precision, sx=suffix)) for i in range(number_of_casts)]
        else:
            file_names = [filename]

        for counter, current_file in enumerate(file_names):
            # We need to account for the crappy date/time columns in the WCO data (which we haven't read in to each
            # self.data attribute).
            num_fields = len(self.variables.names[counter])
            if 'mm_dd_yyyy' in self.variables.names[counter] and 'hh:mm:ss' in self.variables.names[counter]:
                num_fields -= 2
            num_samples = len(getattr(self.data, self.variables.names[counter][0])[counter])

            # For the Header field, there seems to be 12 lines of standard headers plus pairs of lines for each
            # variable.
            bodc_header_lines = 12 + (num_fields * 2)

            start = self.time.datetime[counter].strftime('%Y%m%d%H%M%S')
            lon = np.mean(self.position.lon[counter])
            lat = np.mean(self.position.lat[counter])
            northsouth = 'N'
            westeast = 'E'
            if lat < 0:
                northsouth = 'S'
            if lon < 0:
                westeast = 'W'

            # Some stuff, we'll just ignore/make up for now.
            try:
                sensor = self.position.sensor[counter]
                # If this sensor set is a pair of Nones, then use the depth as below.
                if sensor == [None, None]:
                    sensor = [np.min(self.position.depth[counter]), np.max(self.position.depth[counter])]
            except AttributeError:
                sensor = [np.min(self.position.depth[counter]), np.max(self.position.depth[counter])]

            # The BODC format is... unique. Try to replicate it to the extent which is useful for me. May not be 100%
            # compatible with actual BODC formatting.
            with current_file.open('w') as f:
                # Naturally, we start with three blank lines.
                f.write('\n\n\n')
                f.write('BODC Request Format Std. V1.0           Headers=  {} Data Cycles=   {}\n'.format(bodc_header_lines, num_samples))
                f.write('Series=      {}                     Produced: {}\n'.format('AAAAAAA', datetime.now().strftime('%d-%b-%Y')))  # dummy data for now
                f.write('Id                       AAAAAAAA PML\n')  # more dummy data
                position = '{deglat:03d}d{minlat:.1f}m{hemilat}{deglon:03d}d{minlon:.1f}m{hemilon}'.format(deglat=int(lat),
                                                                                                           minlat=(lat - int(lat)) * 60,
                                                                                                           hemilat=northsouth,
                                                                                                           deglon=int(lon),
                                                                                                           minlon=(lon - int(lon)) * 60,
                                                                                                           hemilon=westeast)
                f.write('{position}                     start:{start}\n'.format(position=position, start=start))
                format_string = 'Dep: floor {depth:.1f} sensor    {sensor_1:.1f}  {sensor_2:.1f} Nom. sample int.:    {interval:.1f} {units}\n'
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
                    if name in ('mm_dd_yyyy', 'hh:mm:ss'):
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
                f.write('(some nonesense for now)\n')
                # A few more new lines in case we're missing some.
                f.write('\n\n\n')
                # The data header.
                f.write('  Cycle     {}\n'.format('   '.join([i[:8] for i in self.variables.names[counter] if i not in ('mm_dd_yyyy', 'hh:mm:ss')])))
                f.write('Number             {}\n'.format('          '.join('f' * num_fields)))
                # Now add the data.
                cycle = ['{})'.format(i) for i in np.arange(num_samples) + 1]
                data = []
                for name in self.variables.names[counter]:
                    # Skip WCO time data columns as we haven't saved those with _ReadData._read_wco().
                    if name in ('mm_dd_yyyy', 'hh:mm:ss'):
                        continue
                    data.append(['{:<0}'.format(i) for i in getattr(self.data, name)[counter]])
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

        def __init__(self, filename):
            """
            Based on file extension, call one of the header parsers below. For now, that's only BODC LST-formatted files.

            Parameters
            ----------
            filename : str, pathlib.Path
                The file name to read in.

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
            self._file = Path(filename)
            self.datetime = None
            self.interval = None
            self.units = None
            self.lon = None
            self.lat = None
            self.depth = None
            self.sensor = [None, None]
            self.header = {}

            if self._file.suffix == '.lst':
                self._read_lst_header()
            elif self._file.suffix == '.txt':
                # I don't like this one bit.
                self._read_wco_header()
            else:
                # Add more readers here.
                pass

        def _read_lst_header(self):
            """
            Get the BODC lst-formatted header. Store the information as a dictionary to make extracting relevant
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
                            self.header['num_records'] = int(line_list[-1])
                        elif line.startswith('Series'):
                            self.header['series_id'] = line_list[1]
                        elif 'start:' in line:
                            raw_position = line_list[0]
                            # Extract degrees, minutes, hemisphere for latitude and longitude.
                            nice_position = [float(raw_position[:3]), float(raw_position[4:7]), raw_position[8],
                                             float(raw_position[9:12]), float(raw_position[13:17]), raw_position[18]]
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
                                self.header['depth'] = float(line_list[2])
                                self.header['sensor'] = [float(i) for i in line_list[4:6]]
                            else:
                                self.header['depth'] = float(line_list[1])
                            self.header['interval'] = float(line_list[-2])
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
                        line_list = split_string(line, separator=';')
                        if line_list[0] == 'DepSM':
                            # We've got a new CTD cast.
                            self.header['num_fields'].append(len(line_list))
                            self.header['record_indices'].append(ctd_counter)
                            # Drop the date/time columns. Also remove illegal characters (specifically /).
                            line_list = [i.replace('/', '_') for i in line_list]
                            self.header['names'].append(line_list)
                            # In order to make the header vaguely usable, grab the initial time and position for this
                            # cast. This means we need to skip a line as we're currently on the header.
                            lon_idx = line_list.index('Longitude')
                            lat_idx = line_list.index('Latitude')
                            date_idx = line_list.index('mm_dd_yyyy')
                            time_idx = line_list.index('hh:mm:ss')
                            # Now we know where to look, extract the relevant information.
                            line = next(f)
                            line_list = split_string(line, separator=';')
                            datetime_string = ' '.join((line_list[date_idx], line_list[time_idx]))
                            self.header['lon'].append(float(line_list[lon_idx]))
                            self.header['lat'].append(float(line_list[lat_idx]))
                            self.header['datetime'].append(datetime.strptime(datetime_string, '%m/%d/%Y %H:%M:%S'))
                            self.header['time_units'].append('datetime')
                            # Some sampling interval. Set to zero as the casts are relatively quick (a few minutes,
                            # tops).
                            self.header['interval'].append(0)
                            self.header['units'].append({i: units[i] for i in self.header['names'][-1] if i in units})
                            self.header['long_name'].append({i: long_name[i] for i in self.header['names'][-1] if i in long_name})

                            # The WCO data have lots of sensors, so just put some placeholders in for the time being.
                            self.header['sensor'].append([None, None])

                            # Manually increment the counter here as we're skipping a line and it would otherwise be
                            # off by one.
                            ctd_counter += 1

            # Get the number of records per cast.
            self.header['num_records'] = np.diff(np.concatenate((self.header['record_indices'], [ctd_counter])))
            # Get the depths for each cast too.
            with self._file.open('r') as f:
                lines = f.readlines()
                for counter, cast_index in enumerate(self.header['record_indices']):
                    offset_index = cast_index - 1  # we want the depth which is the line for the end of the last cast
                    if offset_index > 0:
                        line_list = split_string(lines[offset_index], separator=';')
                        depth_index = self.header['names'][counter].index('DepSM')
                        self.header['depth'].append(float(line_list[depth_index]))
                # Also add the last line's value.
                line_list = split_string(lines[-1], separator=';')
                depth_index = self.header['names'][-1].index('DepSM')
                self.header['depth'].append(float(line_list[depth_index]))

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
                            for name_index, name in enumerate(columns):
                                # Use the cycle number to get the index for the data. Offset the name_index by 1 to
                                # account for the missing Cycle column.
                                if not hasattr(self, name):
                                    setattr(self, name, np.zeros(header['num_records']))
                                data_index = int(line_list[0].replace(')', '')) - 1
                                getattr(self, name)[data_index] = float(line_list[name_index + 1])

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

            # Use the header['record_indices'] and header['num_samples'] to read each CTD cast into a list called
            # self.<variable_name>.
            variable_names = np.unique(header['names'])
            # Remove date/time columns.
            variable_names = [i for i in variable_names if i not in ('mm_dd_yyyy', 'hh_mm_ss')]
            for name in variable_names:
                setattr(self, name, [])

            with file.open('r') as f:
                lines = f.readlines()
                for start, length, names in zip(header['record_indices'], header['num_records'], header['names']):
                    data = lines[start + 1:start + length]
                    data = np.asarray([split_string(i, separator=';') for i in data])
                    if self._debug:
                        print(start, length, start + length + 1)
                        print(data[0, :], data[-1, :])
                    # Dump the data we've got for this cast, excluding the date/time columns.
                    for name in names:
                        if name in ('mm_dd_yyyy', 'hh_mm_ss'):
                            continue
                        getattr(self, name).append(data[:, names.index(name)].astype(float))
                    # Put None in the cumulative list if the current cast is missing a given variable to account for
                    # variables appearing midway through a year.
                    missing_names = set(variable_names) - set(names)
                    for name in missing_names:
                        getattr(self, name).append(None)
