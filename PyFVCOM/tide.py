"""
A series of tools with which tidal data can be extracted from FVCOM NetCDF
model results. Also provides a number of tools to interrogate the SQLite
database of tidal data collated from a range of sources across the north-west
European continental shelf.

"""

from __future__ import print_function

import sys
import scipy
import inspect
import numpy as np

from lxml import etree
from warnings import warn

from PyFVCOM.grid import find_nearest_point
from PyFVCOM.utilities import fix_range, julian_day

try:
    import sqlite3
    use_sqlite = True
except ImportError:
    warn('No sqlite standard library found in this python'
         ' installation. Some functions will be disabled.')
    use_sqlite = False


def add_harmonic_results(db, stationName, constituentName, phase, amplitude, speed, inferred, ident=None, noisy=False):
    """
    Add data to an SQLite database.

    Parameters
    ----------
    db : str
        Full path to an SQLite database. If absent, it will be created.
    stationName : str
        Short name for the current station. This is the table name.
    constituentName : str
        Name of the current tidal constituent being added.
    phase : float
        Tidal constituent phase (in degrees).
    amplitude : float
        Tidal constituent amplitude (in metres).
    speed : float
        Tidal constituent speed (in degrees per hour).
    inferred : str
        'true' or 'false' indicating whether the values are inferred
        (i.e. the time series is too short to perform a robust harmonic
        analysis).
    ident : str
        Optional prefix for the table names in the SQLite database. Usage of
        this option means you can store both u and v data in the same database.
    noisy : bool
        Set to True to enable verbose output.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (add_harmonic_results)'
                           ' is unavailable.')

    if not ident:
        ident = ''
    else:
        ident = '_' + ident

    conn = sqlite3.connect(db)
    c = conn.cursor()

    # Create the necessary tables if they don't exist already
    c.execute('CREATE TABLE IF NOT EXISTS TidalConstituents ( \
        shortName TEXT COLLATE nocase, \
        amplitude FLOAT(10), \
        phase FLOAT(10), \
        speed FLOAT(10), \
        constituentName TEXT COLLATE nocase, \
        amplitudeUnits TEXT COLLATE nocase, \
        phaseUnits TEXT COLLATE nocase, \
        speedUnits TEXT COLLATE nocase, \
        inferredConstituent TEXT COLLATE nocase)')

    if noisy:
        print('amplitude, phase and speed.', end=' ')
    for item in range(len(inferred)):
        c.execute('INSERT INTO TidalConstituents VALUES (?,?,?,?,?,?,?,?,?)',
            (stationName + ident, amplitude[item], phase[item], speed[item], constituentName[item], 'metres', 'degrees', 'degrees per mean solar hour', inferred[item]))

    conn.commit()

    conn.close()


def get_observed_data(db, table, startYear=False, endYear=False, noisy=False):
    """
    Extract the tidal data from the SQLite database for a given station.
    Specify the database (db), the table name (table) which needs to be the
    short name version of the station of interest.

    Optionally supply a start and end year (which if equal give all data from
    that year) to limit the returned data. If no data exists for that station,
    the output is returned as False.

    Parameters
    ----------
    db : str
        Full path to the tide data SQLite database.
    table : str
        Name of the table to be extracted (e.g. 'AVO').
    startYear : bool, optional
        Year from which to start extracting data (inclusive).
    endYear : bool, optional
        Year at which to end data extraction (inclusive).
    noisy : bool, optional
        Set to True to enable verbose output.

    See Also
    --------
    tide.get_observed_metadata : extract metadata for a tide station.

    Notes
    -----
    Search is not fuzzy, so "NorthShields" is not the same as "North Shields".
    Search is case insensitive, however.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_observed_data)'
                           ' is unavailable.')

    if noisy:
        print('Getting data for {} from the database...'.format(table), end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            if startYear and endYear:
                # We've been given a range of data
                if startYear == endYear:
                    # We have the same start and end dates, so just do a
                    # simpler version
                    c.execute('SELECT * FROM {t} WHERE {t}.year == {sy} ORDER BY year, month, day, hour, minute, second'.format(t=table, sy=startYear))
                else:
                    # We have a date range
                    c.execute('SELECT * FROM {t} WHERE {t}.year >= {sy} AND {t}.year <= {ey} ORDER BY year, month, day, hour, minute, second'.format(t=table, sy=startYear, ey=endYear))
            else:
                # Return all data
                c.execute('SELECT * FROM {} ORDER BY year, month, day, hour, minute, second'.format(table))
            # Now get the data in a format we might actually want to use
            data = c.fetchall()

        con.close()

        if noisy:
            print('done.')

    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error {}:'.format(e.args[0]))
            data = [False]

    return data


def get_observed_metadata(db, originator=False, obsdepth=None):
    """
    Extracts the meta data from the supplied database. If the supplied
    originator is False (default), then information from all stations is
    returned.

    Parameters
    ----------
    db : str
        Full path to the tide data SQLite database.
    originator : str, optional
        Specify an originator (e.g. 'NTSLF', 'NSTD', 'REFMAR') to
        extract only that data. Defaults to all data.
    obsdepth : bool, optional
        Set to True to return the observation depth (useful for current meter
        data). Defaults to False.

    Returns
    -------
    lat, lon : list
        Latitude and longitude of the requested station(s).
    site : list
        Short names (e.g. 'AVO' for 'Avonmouth') of the tide stations.
    longName : list
        Long names of the tide stations (e.g. 'Avonmouth').
    depth : list
        If obsdepth=True on input, then depths are returned, otherwise omitted.

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_observed_metadata)'
                           ' is unavailable.')

    con = None
    try:
        con = sqlite3.connect(db)

        c = con.cursor()

        if not originator:
            out = c.execute('SELECT * from Stations where originatorName '
                            'is ? or originatorLongName is ?',
                            [originator, originator])
        else:
            out = c.execute('SELECT * from Stations')

        # Convert it to a set of better formatted values.
        metadata = out.fetchall()
        lat = [float(m[0]) for m in metadata]
        lon = [float(m[1]) for m in metadata]
        site = [str(m[2]) for m in metadata]
        longName = [str(m[3]) for m in metadata]
        if len(metadata) > 4:
            depth = [str(m[4]) for m in metadata]
        else:
            depth = None

    except sqlite3.Error as e:
        if con:
            con.close()
            lat, lon, site, longName, depth = False, False, False, False, False
            raise Exception('SQLite error: {}'.format(e.args[0]))

    if not obsdepth:
        return lat, lon, site, longName
    else:
        return lat, lon, site, longName, depth


def clean_observed_data(data, removeResidual=False):
    """
    Process the observed raw data to a more sensible format. Also
    convert from Gregorian dates to Modified Julian Day (to match FVCOM
    model output times).

    Parameters
    ----------
    data : ndarray
        Array of [YYYY, MM, DD, hh, mm, ss, zeta, flag] data output by
        getObservedData().
    removeResidual : bool, optional
        If True, remove any residual values. Where such data are absent
        (marked by values of -9999 or -99.0), no removal is performed. Defaults
        to False.

    Returns
    -------
    dateMJD : ndarray
        Modified Julian Days of the input data.
    tideDataMSL : ndarray
        Time series of surface elevations from which the mean surface
        elevation has been subtracted. If removeResidual is True, these
        values will omit the atmospheric effects, leaving a harmonic
        signal only.
    npFlagsData : ndarray
        Flag values from the SQLite database (usually -9999, or P, N
        etc. if BODC data).
    allDateTimes : ndarray
        Original date data in [YYYY, MM, DD, hh, mm, ss] format.

    """

    npObsData = []
    npFlagData = []
    for row in data:
        npObsData.append(row[:-1])  # eliminate the flag from the numeric data
        npFlagData.append(row[-1])   # save the flag separately

    # For the tidal data, convert the numbers to floats to avoid issues
    # with truncation.
    npObsData = np.asarray(npObsData, dtype=float)
    npFlagData = np.asarray(npFlagData)

    # Extract the time and tide data
    allObsTideData = np.asarray(npObsData[:, 6])
    allObsTideResidual = np.asarray(npObsData[:, 7])
    allDateTimes = np.asarray(npObsData[:, :6], dtype=float)

    dateMJD = julian_day(allDateTimes, mjd=True)

    # Apply a correction (of sorts) from LAT to MSL by calculating the
    # mean (excluding nodata values (-99 for NTSLF, -9999 for SHOM))
    # and removing that from the elevation.
    tideDataMSL = allObsTideData - np.mean(allObsTideData[allObsTideData > -99])

    if removeResidual:
        # Replace the residuals to remove with zeros where they're -99
        # or -9999 since the net effect at those times is "we don't have
        # a residual, so just leave the original value alone".
        allObsTideResidual[allObsTideResidual <= -99] = 0
        tideDataMSL = tideDataMSL - allObsTideResidual

    return dateMJD, tideDataMSL, npFlagData, allDateTimes


def parse_TAPPY_XML(file):
    """
    Extract values from an XML file created by TAPPY.

    TODO: Allow a list of constituents to be specified when calling
    parse_TAPPY_XML.

    Parameters
    ----------
    file : str
        Full path to a TAPPY output XML file.

    Returns
    -------
    constituentName : list
        Tidal constituent names.
    constituentSpeed : list
        Tidal constituent speeds (in degrees per hour).
    constituentPhase : list
        Tidal constituent phases (in degrees).
    constituentAmplitude : list
        Tidal constituent amplitudes (in metres).
    constituentInference : list
        Flag of whether the tidal constituent was inferred due to a
        short time series for the given constituent.

    """

    tree = etree.parse(open(file, 'r'))

    constituentName = []
    constituentSpeed = []
    constituentInference = []
    constituentPhase = []
    constituentAmplitude = []

    for harmonic in tree.iter('Harmonic'):

        # Still not pretty
        for item in harmonic.iter('name'):
            constituentName.append(item.text)

        for item in harmonic.iter('speed'):
            constituentSpeed.append(item.text)

        for item in harmonic.iter('inferred'):
            constituentInference.append(item.text)

        for item in harmonic.iter('phaseAngle'):
            constituentPhase.append(item.text)

        for item in harmonic.iter('amplitude'):
            constituentAmplitude.append(item.text)

    return constituentName, constituentSpeed, constituentPhase, constituentAmplitude, constituentInference


def get_harmonics(db, stationName, noisy=False):
    """
    Use the harmonics database to extract the results of the harmonic analysis
    for a given station (stationName).

    Parameters
    ----------
    db : str
        Full path to the tidal harmonics SQLite database.
    stationName : str
        Station short name (i.e. table name).
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    siteHarmonics : dict
        Contains all the harmonics data for the given tide station. Keys and units are:
            - 'stationName' (e.g. 'AVO')
            - 'amplitude' (m)
            - 'phase' (degrees)
            - 'speed' (degrees per mean solar hour)
            - 'constituentName' (e.g. 'M2')
            - 'inferredConstituent' ('true'|'false')

    """

    if not use_sqlite:
        raise RuntimeError('No sqlite standard library found in this python'
                           ' installation. This function (get_harmonics) is'
                           ' unavailable.')

    if noisy:
        print('Getting harmonics data for site {}...'.format(stationName), end=' ')

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            c.execute('SELECT * FROM TidalConstituents WHERE shortName = \'' + stationName + '\'')
            data = c.fetchall()

        con.close()
    except sqlite3.Error as e:
        if con:
            con.close()
            print('Error %s:' % e.args[0])
            data = [False]

        if noisy:
            print('extraction failed.')

    # Convert data to a dict of value pairs
    siteHarmonics = {}
    tAmp = np.empty(np.shape(data)[0])
    tPhase = np.empty(np.shape(data)[0])
    tSpeed = np.empty(np.shape(data)[0])
    tConst = np.empty(np.shape(data)[0], dtype="|S7")
    tInfer = np.empty(np.shape(data)[0], dtype=bool)
    for i, constituent in enumerate(data):
        tAmp[i] = constituent[1]
        tPhase[i] = constituent[2]
        tSpeed[i] = constituent[3]
        tConst[i] = str(constituent[4])
        if str(constituent[-1]) == 'false':
            tInfer[i] = False
        else:
            tInfer[i] = True
    siteHarmonics['amplitude'] = tAmp
    siteHarmonics['phase'] = tPhase
    siteHarmonics['speed'] = tSpeed
    siteHarmonics['constituentName'] = tConst
    siteHarmonics['inferredConstituent'] = tInfer

    if noisy:
        print('done.')

    return siteHarmonics


def read_POLPRED(harmonics, noisy=False):
    """
    Load a POLPRED data file into a NumPy array. This can then be used by
    get_harmonics_POLPRED to extract the harmonics at a given loaction, or
    otherwise can be used to simply extract the positions of the POLCOMS grid.

    Parameters
    ----------
    harmonics : str
        Full path to the POLPRED ASCII data file.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    header : dict
        Contains the header data from the POLPRED ASCII file.
    values : ndarray
        Harmonic constituent data formatted as [x, y, nConst * [zZ, zG,
        uZ, uG, vZ, vG]], where nConst is the number of constituents in
        the POLPRED data (15) and z, u and v refer to surface elevation,
        u-vector and v-vector components, respectively. The suffixes Z
        and G refer to amplitude and phase of the z, u and v data.

    See Also
    --------
    tide.grid_POLPRED : Converts the POLPRED data into a rectangular
        gridded data set with values of -999.9 outside the POLPRED domain.

    """

    # Open the harmonics file
    f = open(harmonics, 'r')
    polpred = f.readlines()
    f.close()

    # Read the header into a dict.
    readingHeader = True
    header = {}
    values = []

    if noisy:
        print('Parsing POLPRED raw data...', end=' ')

    for line in polpred:
        if readingHeader:
            if not line.strip():
                # Blank line, which means the end of the header
                readingHeader = False
            else:
                key, parameters = line.split(':')
                header[key.strip()] = parameters.strip()
        else:
            # Remove duplicate whitespaces and split on the resulting
            # single spaces.
            line = line.strip()
            line = ' '.join(line.split())
            values.append(line.split(' '))

    # Make the values into a numpy array
    values = np.asarray(values, dtype=float)

    if noisy:
        print('done.')

    return header, values


def grid_POLPRED(values, noisy=False):
    """
    The POLPRED data are stored as a 2D array, with a single row for each
    location. As such, the lat and long positions are stored in two 1D arrays.
    For the purposes of subsampling, it is much more convenient to have a
    rectangular grid. However, since the POLCOMS model domain is not
    rectangular, it is not possible to simply reshape the POLPRED data.

    To create a rectangular grid, this function builds a lookup table which
    maps locations in the 1D arrays to the equivalent in the 2D array. This is
    achieved as follows:

    1. Create a vector of the unique x and y positions.
    2. Use those positions to search through the 1D array to find the index of
    that position.
    3. Save the 1D index and the 2D indices in a lookup table.
    4. Create a rectangular array whose dimensions match the extent of the
    POLPRED data.
    5. Populate that array with the data, creating a 3D array (x by y by z,
    where z is the number of harmonics).
    6. Use meshgrid to create a rectangular position array (for use with
    pcolor, for example).

    This approach means the grid can be more readily subsampled without the
    need for interpolation (which would change the harmonic constituents).

    Where no data exist (i.e. outside the POLPRED domain), set all values as
    -999.9 (as per POLPRED's land value).

    Parameters
    ----------
    values : ndarray
        Output from read_POLPRED(). See `tide.read_POLPRED'.
    noisy : bool, optional
        Set to True to enable verbose output.

    Returns
    -------
    PX : ndarray
        X values created using np.meshgrid.
    PY : ndarray
        Y values created using np.meshgrid.
    PZ : ndarray
        3D array of harmonic constituent values for the 15 harmonics in
        the POLPRED data at each location in PX and PY. The first two
        dimensions are x and y values (in latitude and longitdue) and
        the third dimension is the amplitude and phases for each of the
        15 constituents for z, u and v data.

    See Also
    --------
    tide.read_POLPRED : Reads in the POLPRED ASCII data.
    tide.get_harmonics_POLPRED : Extract tidal harmonics within a
        threshold distance of a supplied coordinate.

    """

    # Create rectangular arrays of the coordinates in the POLCOMS domain.
    px = np.unique(values[:, 1])
    py = np.unique(values[:, 0])
    PX, PY = np.meshgrid(px, py)

    # I think appending to a list is faster than a NumPy array.
    arridx = []
    for i, (xx, yy) in enumerate(values[:, [1, 0]]):
        if noisy:
            # Only on the first, last and every 1000th line.
            if i == 0 or np.mod(i + 1, 1000) == 0 or i == values[:, 0].shape[0] - 1:
                print('{} of {}'.format(i + 1, np.shape(values)[0]))
        arridx.append([i, px.tolist().index(xx), py.tolist().index(yy)])

    # Now use the lookup table to get the values out of values and into PZ.
    PZ = np.ones([np.shape(py)[0], np.shape(px)[0], np.shape(values)[-1]]) * -999.9
    for idx, xidx, yidx in arridx:
        # Order is the other way around in arridx.
        PZ[yidx, xidx, :] = values[idx, :]

    return PX, PY, PZ


def get_harmonics_POLPRED(harmonics, constituents, lon, lat, stations, noisy=False, distThresh=0.5):
    """
    Function to extract the given constituents at the positions defined
    by lon and lat from a given POLPRED text file.

    The supplied list of names for the stations will be used to generate a
    dict whose structure matches that I've used in the plot_harmonics.py
    script.

    Parameters
    ----------
    harmonics : str
        Full path to the POLPRED ASCII harmonics data.
    constituents : list
        List of tidal constituent names to extract (e.g. ['M2', 'S2']).
    lon, lat : ndarray
        Longitude and latitude positions to find the closest POLPRED
        data point. Uses grid.find_nearest_point to identify the
        closest point. See distThresh below.
    stations : list
        List of tide station names (or coordinates) which are used as
        the keys in the output dict.
    noisy : bool, optional
        Set to True to enable verbose output.
    distThresh : float, optional
        Give a value (in the units of lon and lat) which limits the
        distance to which POLPRED points are returned. Essentially gives
        an upper threshold distance beyond which a point is considered
        not close enough.

    Returns
    -------
    out : dict
        A dict whose keys are the station names. Within each of those
        dicts is another dict whose keys are 'amplitude', 'phase' and
        'constituentName'.
        In addition to the elevation amplitude and phases, the u and v
        amplitudes and phases are also extracted into the dict, with the
        keys 'uH', 'vH', 'uG' and 'vG'.
        Finally, the positions from the POLPRED data is stored with the
        keys 'latitude' and 'longitude'. The length of the arrays within
        each of the secondary dicts is dependent on the number of
        constituents requested.

    See Also
    --------
    tide.read_POLPRED : Read in the POLPRED data to split the ASCII
        file into a header dict and an ndarray of values.
    grid.find_nearest_point : Find the closest point in one set of
        coordinates to a specified point or set of points.

    """

    header, values = read_POLPRED(harmonics, noisy=noisy)

    # Find the nearest points in the POLCOMS grid to the locations
    # requested.
    nearestX, nearestY, distance, index = find_nearest_point(values[:, 1],
                                                             values[:, 0],
                                                             lon,
                                                             lat,
                                                             maxDistance=distThresh)

    # Get a list of the indices from the header for the constituents we're
    # extracting.
    ci = np.empty([np.shape(constituents)[0], 6], dtype=int)
    for i, con in enumerate(constituents):
        tmp = header['Harmonics'].split(' ').index(con)
        # Times 6 because of the columns per constituent
        ci[i, :] = np.repeat(tmp * 6, 6)
        # Add the offsets for the six harmonic components (amplitude and phase
        # of z, u and v).
        ci[i, :] = ci[i, :] + np.arange(6)

    # Plus 3 because of the lat, long and flag columns.
    ci += 3

    # Make a dict of dicts for each station supplied.
    out = {}

    # Find the relevant data for the current site.
    for c, key in enumerate(stations):
        if noisy:
            print('Extracting site {}...'.format(key), end=' ')
            sys.stdout.flush()

        data = {}
        if np.isnan(index[c]):
            if noisy:
                print('skipping (outside domain).')
        else:
            keys = ['amplitude', 'phase', 'uH', 'ug', 'vH', 'vg']
            for n, val in enumerate(keys):
                data[val] = values[index[c], ci[:, n]]

            data['constituentName'] = constituents
            data['latitude'] = values[index[c], 0]
            data['longitude'] = values[index[c], 1]

            out[key] = data

            if noisy:
                print('done.')
                sys.stdout.flush()

    return out


def make_water_column(zeta, h, siglay):
    """
    Make a time varying water column array with the surface elevation at the
    surface and depth negative down.

    Parameters
    ----------
    siglay : ndarray
        Sigma layers [lay, nodes]
    h : ndarray
        Water depth [nodes] or [time, nodes]
    zeta : ndarray
        Surface elevation [time, nodes]

    Returns
    -------
    depth : ndarray
        Time-varying water depth (with the surface depth varying rather than
        the seabed) [time, lay, nodes].

    Todo
    ----
    Tidy up the try/excepth block with an actual error.

    """

    # Fix the range of siglay to be -1 to 0 so we don't get a wobbly seabed.
    siglay = fix_range(siglay, -1, 0)

    # We may have a single node, in which case we don't need the newaxis,
    # otherwise, we do.
    try:
        z = (zeta + h) * -siglay
    except:
        z = (zeta + h)[:, np.newaxis, :] * -siglay[np.newaxis, ...]

    try:
        z = z - h
    except ValueError:
        z = z - h[:, np.newaxis, :]

    return z


class Lanczos:
    """
    Create a Lanczos filter object with specific parameters. Pass a time series to filter() to apply that filter to
    the time series.

    Notes
    -----
    This is a python reimplementation of the MATLAB lanczosfilter.m function from
    https://mathworks.com/matlabcentral/fileexchange/14041.

    NaN values are replaced by the mean of the time series and ignored. If you have a better idea, just let me know.

    Reference
    ---------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical Oceanography". Elsevier, 2d ed.,
    2004. On pages 533-539.

    """
    def __init__(self, dt=1, cutoff=None, samples=100, passtype='low'):
        """

        Parameters
        ----------
        dt : float, optional
            Sampling interval. Defaults to 1. (dT in the MATLAB version).
        cutoff : float, optional
            Cutoff frequency in minutes at which to pass data. Defaults to the half the Nyquist frequency. (Cf in the MATLAB version).
        samples : int, optional
            Number of samples in the window. Defaults to 100. (M in the MATLAB version)
        passtype : str
            Set the filter to `low' to low-pass (default) or `high' to high-pass. (pass in the MATLAB version).

        """

        self.dt = dt
        self.cutoff = cutoff
        self.samples = samples
        self.passtype = passtype

        if self.passtype == 'low':
            filterindex = 0
        elif self.passtype == 'high':
            filterindex = 1
        else:
            raise ValueError("Specified `passtype' is invalid. Select `high' or `low'.")

        # Nyquist frequency
        self.nyquist_frequency = 1 / (2 * self.dt)
        if not self.cutoff:
            cutoff = self.nyquist_frequency / 2

        # Normalize the cut off frequency with the Nyquist frequency:
        self.cutoff = self.cutoff / self.nyquist_frequency

        # Lanczos cosine coefficients:
        self._lanczos_filter_coef()
        self.coef = self.coef[:, filterindex]

    def _lanczos_filter_coef(self):
        # Positive coefficients of Lanczos [low high]-pass.
        hkcs = self.cutoff * np.array([1] + (np.sin(np.pi * np.linspace(1, self.samples, self.samples) * self.cutoff) / (np.pi * np.linspace(1, self.samples, self.samples) * self.cutoff)).tolist())
        sigma = sigma = np.array([1] + (np.sin(np.pi * np.linspace(1, self.samples, self.samples) / self.samples) / (np.pi * np.linspace(1, self.samples, self.samples) / self.samples)).tolist())
        hkB = hkcs * sigma
        hkA = -hkB
        hkA[0] = hkA[0] + 1

        self.coef = np.array([hkB.ravel(), hkA.ravel()]).T

    def _spectral_window(self):
        # Window of cosine filter in frequency space.
        eps = np.finfo(np.float32).eps
        self.Ff = np.arange(0, 1 + eps, 2 / self.N)  # add an epsilon to enclose the stop in the range.
        self.window = np.zeros(len(self.Ff))
        for i in range(len(self.Ff)):
            self.window[i] = self.coef[0] + 2 * np.sum(self.coef[1:] * np.cos((np.arange(1, len(self.coef))) * np.pi * self.Ff[i]))

    def _spectral_filtering(self, x):
        # Filtering in frequency space is multiplication, (convolution in time space).
        Cx = scipy.fft(x.ravel())
        Cx = Cx[:(self.N // 2) + 1]
        CxH = Cx * self.window.ravel()
        # Mirror CxH and append it to itself, dropping the values depending on the length of the input.
        CxH = np.concatenate((CxH, scipy.conj(CxH[1:self.N - len(CxH) + 1][::-1])))
        y = np.real(scipy.ifft(CxH))

        return y

    def filter(self, x):
        """
        Filter the given time series values and return the filtered data.

        Parameters
        ----------
        x : np.ndarray
            Time series values (1D).

        Returns
        -------
        y : np.ndarray
            Filtered time series values (1D).

        """

        # Filter in frequency space:
        self.N = len(x)
        self._spectral_window()
        self.Ff *= self.nyquist_frequency

        # Replace NaNs with the mean (ideas?):
        inan = np.isnan(x)
        if np.any(inan):
            xmean = np.nanmean(x)
            x[inan] = xmean

        # Filtering:
        y = self._spectral_filtering(x)

        # Make sure we've got arrays which match in size.
        if not (len(x) == len(y)):
            raise ValueError('Hmmmm. Fix the arrays!')

        return y


def lanczos(x, dt=1, cutoff=None, samples=100, passtype='low'):
    """
    Apply a Lanczos low- or high-pass filter to a time series.

    Parameters
    ----------
    x : np.ndarray
        1-D times series values.
    dt : float, optional
        Sampling interval. Defaults to 1. (dT in the MATLAB version).
    cutoff : float, optional
        Cutoff frequency in minutes at which to pass data. Defaults to the half the Nyquist frequency. (Cf in the MATLAB version).
    samples : int, optional
        Number of samples in the window. Defaults to 100. (M in the MATLAB version)
    passtype : str
        Set the filter to `low' to low-pass (default) or `high' to high-pass. (pass in the MATLAB version).

    Returns
    -------
    y : np.ndarray
        Smoothed time series.
    coef : np.ndarray
        Coefficients of the time window (cosine)
    window : np.ndarray
        Frequency window (aprox. ones for Ff lower(greater) than Fc if low(high)-pass filter and ceros otherwise)
    Cx : np.ndarray
        Complex Fourier Transform of X for Ff frequencies
    Ff : np.ndarray
        Fourier frequencies, from 0 to the Nyquist frequency.

    Notes
    -----
    This is a python reimplementation of the MATLAB lanczosfilter.m function from
    https://mathworks.com/matlabcentral/fileexchange/14041.

    NaN values are replaced by the mean of the time series and ignored. If you have a better idea, just let me know.

    Reference
    ---------
    Emery, W. J. and R. E. Thomson. "Data Analysis Methods in Physical Oceanography". Elsevier, 2d ed.,
    2004. On pages 533-539.

    """

    if passtype == 'low':
        filterindex = 0
    elif passtype == 'high':
        filterindex = 1
    else:
        raise ValueError("Specified `passtype' is invalid. Select `high' or `low'.")

    # Nyquist frequency
    nyquist_frequency = 1 / (2 * dt)
    if not cutoff:
        cutoff = nyquist_frequency / 2

    # Normalize the cut off frequency with the Nyquist frequency:
    cutoff = cutoff / nyquist_frequency

    # Lanczos cosine coefficients:
    coef = _lanczos_filter_coef(cutoff, samples)
    coef = coef[:, filterindex]

    # Filter in frequency space:
    window, Ff = _spectral_window(coef, len(x))
    Ff = Ff * nyquist_frequency

    # Replace NaNs with the mean (ideas?):
    inan = np.isnan(x)
    if np.any(inan):
        xmean = np.nanmean(x)
        x[inan] = xmean

    # Filtering:
    y, Cx = _spectral_filtering(x, window)

    # Make sure we've got arrays which match in size.
    if not (len(x) == len(y)):
        raise ValueError('Hmmmm. Fix the arrays!')

    return y, coef, window, Cx, Ff


def _lanczos_filter_coef(cutoff, samples):
    # Positive coefficients of Lanczos [low high]-pass.
    hkcs = cutoff * np.array([1] + (np.sin(np.pi * np.linspace(1, samples, samples) * cutoff) / (np.pi * np.linspace(1, samples, samples) * cutoff)).tolist())
    sigma = sigma = np.array([1] + (np.sin(np.pi * np.linspace(1, samples, samples) / samples) / (np.pi * np.linspace(1, samples, samples) / samples)).tolist())
    hkB = hkcs * sigma
    hkA = -hkB
    hkA[0] = hkA[0] + 1
    coef = np.array([hkB.ravel(), hkA.ravel()]).T

    return coef


def _spectral_window(coef, N):
    # Window of cosine filter in frequency space.
    eps = np.finfo(np.float32).eps
    Ff = np.arange(0, 1 + eps, 2 / N)  # add an epsilon to enclose the stop in the range.
    window = np.zeros(len(Ff))
    for i in range(len(Ff)):
        window[i] = coef[0] + 2 * np.sum(coef[1:] * np.cos((np.arange(1, len(coef))) * np.pi * Ff[i]))

    return window, Ff


def _spectral_filtering(x, window):
    # Filtering in frequency space is multiplication, (convolution in time space).
    Nx = len(x)
    Cx = scipy.fft(x.ravel())
    Cx = Cx[:(Nx // 2) + 1]
    CxH = Cx * window.ravel()
    # Mirror CxH and append it to itself, dropping the values depending on the length of the input.
    CxH = np.concatenate((CxH, scipy.conj(CxH[1:Nx-len(CxH)+1][::-1])))
    y = np.real(scipy.ifft(CxH))
    return y, Cx


# Add for backwards compatibility.
def julianDay(*args, **kwargs):
    warn('{} is deprecated. Use julian_day instead.'.format(inspect.stack()[0][3]))
    return julian_day(*args, **kwargs)


def gregorianDate(*args, **kwargs):
    warn('{} is deprecated. Use gregorian_date instead.'.format(inspect.stack()[0][3]))
    return gregorian_date(*args, **kwargs)


def addHarmonicResults(*args, **kwargs):
    warn('{} is deprecated. Use add_harmonic_results instead.'.format(inspect.stack()[0][3]))
    return add_harmonic_results(*args, **kwargs)


def getObservedData(*args, **kwargs):
    warn('{} is deprecated. Use get_observed_data instead.'.format(inspect.stack()[0][3]))
    return get_observed_data(*args, **kwargs)


def getObservedMetadata(*args, **kwargs):
    warn('{} is deprecated. Use get_observed_metadata instead.'.format(inspect.stack()[0][3]))
    return get_observed_metadata(*args, **kwargs)


def cleanObservedData(*args, **kwargs):
    warn('{} is deprecated. Use clean_observed_data instead.'.format(inspect.stack()[0][3]))
    return clean_observed_data(*args, **kwargs)


def parseTAPPyXML(*args, **kwargs):
    warn('{} is deprecated. Use parse_TAPPY_XML instead.'.format(inspect.stack()[0][3]))
    return parse_TAPPY_XML(*args, **kwargs)


def getHarmonics(*args, **kwargs):
    warn('{} is deprecated. Use get_harmonics instead.'.format(inspect.stack()[0][3]))
    return get_harmonics(*args, **kwargs)


def readPOLPRED(*args, **kwargs):
    warn('{} is deprecated. Use read_POLPRED instead.'.format(inspect.stack()[0][3]))
    return read_POLPRED(*args, **kwargs)


def gridPOLPRED(*args, **kwargs):
    warn('{} is deprecated. Use grid_POLPRED instead.'.format(inspect.stack()[0][3]))
    return grid_POLPRED(*args, **kwargs)


def getHarmonicsPOLPRED(*args, **kwargs):
    warn('{} is deprecated. Use get_harmonics_POLPRED instead.'.format(inspect.stack()[0][3]))
    return get_harmonics_POLPRED(*args, **kwargs)
