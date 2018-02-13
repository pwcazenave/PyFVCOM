"""
Functions to interrogate and extract CTD data from the ctd.db SQLite3 database.

"""

from __future__ import print_function

import inspect
from pathlib import Path
from warnings import warn
from datetime import datetime

import numpy as np

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


class BODC(object):
    """ Work with BODC data. Mainly focused around CTD data. """

    def __init__(self):
        """ Initialise a BODC object. """

        self.debug = False

        self.time = type('time', (), {})()
        self.data = type('data', (), {})()
        self.position = type('position', (), {})()
        self.metadata = type('metadata', (), {})()  # maybe one day this will be used.

    def _read_lst(self, filename):
        """
        Read the given LST-formatted file and process the data accordingly.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.

        Provides
        --------
        self.data, self.time and self.position are populated accordingly.

        """

        # Initialise things which we expect to exist later. For example, self.position.sensor needn't exist,
        # but if it gets created, then it's used when writing out a file.
        self.data.names = []
        self.data.units = {}
        self.data.long_name = {}
        self.position.lon = None
        self.position.lat = None
        self.position.depth = None

        # Make something to tidy up the lines.
        clean = lambda x: [i.strip() for i in line.split(' ') if i.strip()]

        filename = Path(filename)
        with filename.open('r') as f:
            for line in f:
                line = line.strip()
                if line:
                    line_list = clean(line)
                    if self.debug:
                        print('headers {}'.format(line))
                    if line.startswith('BODC'):
                        header_length = int(line_list[6])
                        num_records = int(line_list[-1])
                    elif line.startswith('Series'):
                        name = line_list[1]
                    elif 'start:' in line:
                        raw_position = line_list[0]
                        # Extract degrees, minutes, hemisphere for latitude and longitude.
                        nice_position = [float(raw_position[:3]), float(raw_position[4:7]), raw_position[8],
                                         float(raw_position[9:12]), float(raw_position[13:17]), raw_position[18]]
                        self.position.lon = nice_position[3] + (nice_position[4] / 60)
                        if nice_position[5] == 'W':
                            self.position.lon = -self.position.lon
                        self.position.lat = nice_position[0] + (nice_position[1] / 60)
                        if nice_position[2] == 'S':
                            self.position.lat = -self.position.lat

                        start = line_list[1].split(':')[1]
                        self.time.datetime = [datetime.strptime(start, '%Y%m%d%H%M%S')]
                    elif line.startswith('Dep'):
                        if 'floor' in line:
                            self.position.depth = float(line_list[2])
                            self.position.sensor = [float(i) for i in line_list[4:6]]
                        else:
                            self.position.depth = float(line_list[1])
                        self.time.interval = float(line_list[-2])
                        self.time.units = line_list[-1]
                    elif 'included' in line:
                        # Get the number of parameters in this file.
                        num_fields = int(line_list[0])
                        # Read one more line here and then break out here as we handle reading the metadata in a
                        # separate loop.
                        next(f)
                        break

            # Extract the information about each variable.
            for record in range(num_fields):
                line = next(f).strip()
                if self.debug:
                    print('metadata {}'.format(line))
                if line:
                    line_list = clean(line)
                    self.data.names.append(line_list[0])
                    # Preallocate the data array ready for the data.
                    setattr(self.data, self.data.names[-1], np.zeros((num_records)))
                    self.data.units[self.data.names[-1]] = line_list[-1]
                    # Skip to the next line now so we can get the long_name value.
                    line = next(f).strip()
                    self.data.long_name[self.data.names[-1]] = line

            # Now ignore the rest until we hit the data itself.
            at_data = False
            for line in f:
                line = line.strip()
                if self.debug:
                    print('data {}'.format(line))
                line_list = clean(line)
                if line:
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
                            data_index = int(line_list[0].replace(')', '')) - 1
                            getattr(self.data, name)[data_index] = float(line_list[name_index + 1])

    def _read_qxf(self, filename):
        """
        Read the given QXF-formatted file and process the data accordingly.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.

        Provides
        --------
        self.data, self.time and self.position are populated accordingly.

        """
        pass

    def read(self, filename):
        """
        Generic wrapper around the two main BODC file formats I'm interested in (QXF. vs LST, essentially netCDF vs.
        ASCII).

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to read in.

        Provides
        --------
        self.data, self.time and self.position are populated accordingly.

        """
        try:
            suffix = Path(filename).suffix
        except ValueError:
            suffix = filename.suffix

        if suffix == '.qxf':
            self._read_qxf(filename)
        elif suffix == '.lst':
            self._read_lst(filename)
        else:
            raise ValueError('Unsupported file extension: {}, supply either .lst or .qxf'.format(suffix))

    def _write_lst(self, filename):
        """
        Write the given data to LST-formatted file.

        Parameters
        ----------
        filename : str, pathlib.Path
            The file name to write out to.

        """

        # Parse the data for relevant metadata.
        num_fields = len(self.data.names)
        num_samples = len(getattr(self.data, self.data.names[0]))

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
            f.write('BODC Request Format Std. V1.0           Headers=  {} Data Cycles=   {}\n'.format(14, num_samples))
            f.write('Series=      {}                     Produced: {}\n'.format('AAAAAAAA', datetime.now().strftime('%d-%b-%Y')))  # dummy data for now
            f.write('Id                       AAAAAAAA PML\n')  # more dummy data
            position = '{deglat:03d}d{minlat:.1f}m{hemilat}{deglon:03d}d{minlon:.1f}m{hemilon}'.format(deglat=int(lat),
                                                                                                       minlat=lat - int(lat),
                                                                                                       hemilat=northsouth,
                                                                                                       deglon=int(lon),
                                                                                                       minlon=lon - int(lon),
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
            for name in self.data.names:
                header_string = '{name:8s}  {f} {P} {Q} {missing:.2f} {min:.2f} {max:.2f} {unit}\n'
                f.write(header_string.format(name=name[:8],
                                             f=0,
                                             P=0,
                                             Q=0,
                                             missing=-1,
                                             min=np.min(getattr(self.data, name)),
                                             max=np.max(getattr(self.data, name)),
                                             unit=self.data.units[name]))
                f.write('{}\n'.format(self.data.long_name[name][:80]))  # maximum line length is 80 characters
            f.write('\n')
            f.write('Format Record\n')
            f.write('(some nonesense for now)\n')
            # A few more new lines in case we're missing some.
            f.write('\n\n\n')
            # The data header.
            f.write('  Cycle {}\n'.format('  '.join([i[:8] for i in self.data.names])))
            f.write('Number          {}\n'.format('       '.join('f' * num_fields)))
            # Now add the data.
            cycle = np.arange(num_samples) + 1
            data = []
            for name in self.data.names:
                data.append(getattr(self.data, name))
            # data = np.column_stack((cycle, np.asarray(data)))

            # Must be possible to dump a whole numpy array in one shot...
            # np.savetxt(f, data, fmt='%f')
            # f.writelines(data)


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
