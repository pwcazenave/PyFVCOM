"""
Functions to interrogate and extract CTD data from the ctd.db SQLite3 database.

"""

from __future__ import print_function

import inspect
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

        self.debug = False
        self.data = None  # assume we're not reading data unless self.read() is called.

        self.file = Path(filename)
        # Read the header into self so we can pass it around more easily. Really, having the header as a dictionary
        # is probably the most sensible thing.
        self.header = self._ParseHeader(self.file)
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

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to which to write out.

        """

        # Parse the data for relevant metadata.
        num_fields = len(self.variables.names)
        num_samples = len(getattr(self.data, self.variables.names[0]))

        # For the Header field, there seems to be 12 lines of standard headers plus pairs of lines for each
        # variable.
        bodc_header_lines = 12 + (num_fields * 2)

        start = np.min(self.time.datetime).strftime('%Y%m%d%H%M%S')
        lon = np.mean(self.position.lon)
        lat = np.mean(self.position.lat)
        northsouth = 'N'
        westeast = 'E'
        if lat < 0:
            northsouth = 'S'
        if lon < 0:
            westeast = 'W'

        # Some stuff, we'll just ignore/make up for now.
        try:
            sensor = self.position.sensor
        except AttributeError:
            sensor = [np.min(self.position.depth), np.max(self.position.depth)]

        # The BODC format is... unique. Try to replicate it to the extent which is useful for me. May not be 100%
        # compatible with actual BODC formatting.
        filename = Path(filename)
        with filename.open('w') as f:
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
            f.write(format_string.format(depth=self.position.depth,
                                         sensor_1=sensor[0],
                                         sensor_2=sensor[1],
                                         interval=self.time.interval,
                                         units=self.time.units))
            f.write('    {} Parameters included:\n'.format(num_fields))
            # Now we're into the headers for each variable.
            f.write('Parameter f    P    Q Absent Data Value Minimum Value  Maximum Value       Units\n')
            for name in self.variables.names:
                header_string = '{name:8s}  {f} {P} {Q} {missing:.2f} {min:.2f} {max:.2f} {unit}\n'
                f.write(header_string.format(name=name[:8],
                                             f='Y',
                                             P=0,
                                             Q=0,
                                             missing=-1,
                                             min=np.min(getattr(self.data, name)),
                                             max=np.max(getattr(self.data, name)),
                                             unit=self.variables.units[name]))
                f.write('{}\n'.format(self.variables.long_name[name][:80]))  # maximum line length is 80 characters
            f.write('\n')
            f.write('Format Record\n')
            f.write('(some nonesense for now)\n')
            # A few more new lines in case we're missing some.
            f.write('\n\n\n')
            # The data header.
            f.write('  Cycle     {}\n'.format('   '.join([i[:8] for i in self.variables.names])))
            f.write('Number             {}\n'.format('          '.join('f' * num_fields)))
            # Now add the data.
            cycle = ['{})'.format(i) for i in np.arange(num_samples) + 1]
            data = []
            for name in self.variables.names:
                data.append(['{:<0}'.format(i) for i in getattr(self.data, name)])
            data = np.column_stack((cycle, np.asarray(data).T))

        # Must be possible to dump a whole numpy array in one shot...
        with filename.open('ab') as f:
            np.savetxt(f, data, fmt=['%10s'] * (num_fields + 1))

    def write_qxf(self, filename):
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
            self.file = Path(filename)
            self.datetime = None
            self.interval = None
            self.units = None
            self.lon = None
            self.lat = None
            self.depth = None
            self.sensor = [None, None]
            self.header = {}

            if self.file.suffix == '.lst':
                self._read_bodc_header()
            else:
                # Add more readers here.
                pass

        def _read_bodc_header(self):
            """
            Get the BODC header. Store the information as a dictionary to make extracting relevant information easier.

            Provides
            --------
            header : dict
                Dictionary of the header information in the BODC LST-formatted file. This includes a standard set of keys
                plus those defined in the header itself. The standard keys are:
                    'file_name' - the file name from which we've read.
                    'names' - the variable names in the file.
                    'units' - the variable units.
                    'long_name' - the variable descriptions.

            """

            self.header['file_name'] = str(self.file)  # keep a record of the file we're opening.
            self.header['names'] = []  # store the variable names
            self.header['units'] = {}  # the variables' units
            self.header['long_name'] = {}  # the variable descriptions

            with self.file.open('r') as f:
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
                            self.header['longitude'] = nice_position[3] + (nice_position[4] / 60)
                            if nice_position[5] == 'W':
                                self.header['longitude'] = -self.header['longitude']
                            self.header['latitude'] = nice_position[0] + (nice_position[1] / 60)
                            if nice_position[2] == 'S':
                                self.header['latitude'] = -self.header['latitude']

                            start = line_list[1].split(':')[1]
                            self.header['time'] = [datetime.strptime(start, '%Y%m%d%H%M%S')]
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

            """

            header : dict

            """


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
            if Path(header['file_name']).suffix == '.lst':
                self._read_lst(header)
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
