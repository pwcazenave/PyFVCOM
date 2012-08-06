"""
Aggregate the tidal database tools into a single file instead of having them
strewn all over the place.

"""

def julianDay(gregorianDateTime, mjd=False):
    """
    For a given gregorian date format (YYYY,MM,DD,hh,mm,ss) get the Julian Day.
    hh,mm,ss are optional, and zero if omitted (i.e. midnight).

    Output array precision is the same as input precision, so if you want
    sub-day precision, make sure your input data are floats.

    Julian Day epoch: 12:00 January 1, 4713 BC, Monday
    Modified Julain Day epoch: 00:00 November 17, 1858, Wednesday

    mjd=True applies the offset from Julian Day to Modified Julian Day.

    Modified after code at http://paste.lisp.org/display/73536 and
    http://home.online.no/~pjacklam/matlab/software/util/timeutil/date2jd.m

    """

    try:
        import numpy as np
    except ImportError:
        raise ImportError('Failed to import NumPy')

    try:
        nr, nc = np.shape(gregorianDateTime)
    except:
        nc = np.shape(gregorianDateTime)
        nr = 1

    if nc < 6:
        # We're missing some aspect of the time. Let's assume it's the least
        # significant value (i.e. seconds first, then minutes, then hours).
        # Set missing values to zero.
        numMissing = 6 - nc
        if numMissing > 0:
            extraCols = np.zeros([nr,numMissing])
            gregorianDateTime = np.hstack([gregorianDateTime, extraCols])

    if nr > 1:
        year = gregorianDateTime[:,0]
        month = gregorianDateTime[:,1]
        day = gregorianDateTime[:,2]
        hour = gregorianDateTime[:,3]
        minute = gregorianDateTime[:,4]
        second = gregorianDateTime[:,5]
    else:
        year = gregorianDateTime[0]
        month = gregorianDateTime[1]
        day = gregorianDateTime[2]
        hour = gregorianDateTime[3]
        minute = gregorianDateTime[4]
        second = gregorianDateTime[5]

    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + (12 * a) - 3
    # Updated the day fraction based on MATLAB function:
    #   http://home.online.no/~pjacklam/matlab/software/util/timeutil/date2jd.m
    jd = day + (( 153 * m + 2) // 5) \
        + y * 365 + (y // 4) - (y // 100) + (y // 400) - 32045 \
        + (second + 60 * minute + 3600 * (hour - 12)) / 86400

    if mjd:
        return jd - 2400000.5
    else:
        return jd

def addHarmonicResults(db, stationName, constituentName, phase, amplitude, speed, inferred, noisy=False):
    """
    Add data to an SQLite database.

    - db specifies an SQLite databse. If it doesn't exist, it will be created.
    - stationName is the short name (i.e. AVO not Avonmouth)
    - constituent Name is M2, S2 etc.
    - phase is in degrees
    - amplitude is in metres
    - speed is in degrees/hour
    - inferred is 'true' or 'false' (as strings, not python special values)

    Optionally specify noisy=True to turn on verbose output.

    """

    try:
        import sqlite3
    except ImportError:
        raise ImportError('Failed to import the SQLite3 module')

    conn = sqlite3.connect(db)
    c = conn.cursor()


    # Create the necessary tables if they don't exist already
    c.execute('CREATE TABLE IF NOT EXISTS TidalConstituents (\
        shortName TEXT COLLATE nocase,\
        amplitude FLOAT(10),\
        phase FLOAT(10),\
        speed FLOAT(10),\
        constituentName TEXT COLLATE nocase,\
        amplitudeUnits TEXT COLLATE nocase,\
        phaseUnits TEXT COLLATE nocase,\
        speedUnits TEXT COLLATE nocase,\
        inferredConstituent TEXT COLLATE nocase\
        )')

    if noisy:
        print 'amplitude, phase and speed.',
    for item in xrange(len(inferred)):
        c.execute('INSERT INTO TidalConstituents VALUES (?,?,?,?,?,?,?,?,?)',\
            (stationName, amplitude[item], phase[item], speed[item], constituentName[item], 'metres', 'degrees', 'degrees per mean solar hour', inferred[item]))

    conn.commit()

    conn.close()

def getObservedData(db, table, startYear=False, endYear=False, noisy=False):
    """
    Extract the tidal data from the SQLite database for a given station.
    Specify the database (db), the table name (table) which needs to be the
    short name version of the station of interest.

    Optionally supply a start and end year (which if equal give all data from
    that year) to limit the returned data. If no data exists for that station,
    the output is returned as False.

    Search is not fuzzy, so "NorthShields" is not the same as "North Shields".
    Search is case insensitive, however.

    Add noisy=True to turn on verbose output.

    """

    try:
        import sqlite3
    except ImportError:
        raise ImportError('Failed to import the SQLite3 module')

    if noisy:
        print 'Getting data for %s from the database.' % table

    try:
        con = sqlite3.connect(db)

        with con:
            c = con.cursor()
            if startYear and endYear:
                # We've been given a range of data
                if startYear == endYear:
                    # We have the same start and end dates, so just do a
                    # simpler version
                    c.execute('SELECT * FROM ' + table + ' WHERE ' + \
                    table + '.year == ' + str(startYear) + \
                    ' ORDER BY year, month, day, hour, minute, second')
                else:
                    # We have a date range
                    c.execute('SELECT * FROM ' + table + ' WHERE ' + \
                    table + '.year > ' + str(startYear) + \
                    ' AND ' + table + '.year < ' + str(endYear) + \
                    ' ORDER BY year, month, day, hour, minute, second')
            else:
                # Return all data
                c.execute('SELECT * FROM ' + table + \
                    ' ORDER BY year, month, day, hour, minute, second')
            # Now get the data in a format we might actually want to use
            data = c.fetchall()

        con.close()

    except sqlite3.Error, e:
        if con:
            con.close()
            print 'Error %s:' % e.args[0]
            data = [False]

    return data

def getObservedMetadata(db, originator=False):
    """
    Extracts the meta data from the tidal elevations database. If the supplied
    originator is False (default), then information from all stations is
    returned.

    """

    try:
        import sqlite3
    except ImportError:
        raise ImportError('Failed to import the SQLite3 module')

    try:
        con = sqlite3.connect(db)

        c = con.cursor()

        if originator is not False:
            out = c.execute('SELECT * from Stations where originatorName is ? or originatorLongName is ?',\
                [originator, originator])
        else:
            out = c.execute('SELECT * from Stations')

        # Convert it to a set of better formatted values.
        metadata = out.fetchall()
        lat = [float(m[0]) for m in metadata]
        lon = [float(m[1]) for m in metadata]
        site = [str(m[2]) for m in metadata]
        longName = [str(m[3]) for m in metadata]

    except sqlite3.Error, e:
        if con:
            con.close()
            print 'Error %s:' % e.args[0]
            lat, lon, site, longName = [False, False, False, False]

    return lat, lon, site, longName

def cleanObservedData(data):
    """
    Take the output of getObservedData and identify NaNs (with a mask). Also
    convert times into Modified Julian Date.

    Tidal elevations also have the mean tidal elevation for the time series
    removed from all values. This is a sort of poor man's correction to mean
    sea level.

    """

    try:
        import numpy as np
    except:
        raise ImportError('Failed to import NumPy')

    npObsData = []
    npFlagData = []
    for row in data:
        npObsData.append(row[0:-1]) # eliminate the flag from the numeric data
        npFlagData.append(row[-1]) # save the flag separately

    # For the tidal data, convert the numbers to floats to avoid issues
    # with truncation.
    npObsData = np.asarray(npObsData, dtype=float)
    npFlagData = np.asarray(npFlagData)

    # Extract the time and tide data
    allObsTideData = np.asarray(npObsData[:,6])
    allDateTimes = np.asarray(npObsData[:,0:6])

    dateMJD = julianDay(allDateTimes, mjd=True)

    # Apply a correction (of sorts) from LAT to MSL by calculating the
    # mean (excluding nodata values (-99 for NTSLF, -9999 for SHOM))
    # and removing that from the elevation.
    tideDataMSL = allObsTideData - np.mean(allObsTideData[allObsTideData>-99])

    return dateMJD, tideDataMSL, npFlagData

def runTAPPy(data, sparseDef=False, noisy=False, deleteFile=True):
    """
    A simple wrapper to perform a harmonic analysis on the supplied data.
    Input data format is YYYY, MM, DD, hh, mm, ss, ZZ as a numpy array.

    TAPPy is called as follows:

        tappy.py analysis --def_filename=sparse.def --outputxml=tempfile.xml --quiet tempinput.txt

    The output XML file is parsed with parseTAPPyXML to return a series of
    variables containing the analysis output. The input file tempinput.txt is
    deleted once the analysis is complete, unless deleteFile is set to False,
    in which case it is left where it is. To find it, pass noisy=True to be
    given more verbose output.

    The default sparse definition file is:

        /users/modellers/pica/Data/proc/tides/sparse.def

    Pass an alternate value to sparseDef to use a different one.

    """

    try:
        import subprocess
    except ImportError:
        raise ImportError('Failed to import the subprocess module')

    try:
        import tempfile
    except ImportError:
        raise ImportError('Failed to import the tempfile module')

    try:
        import numpy as np
    except:
        raise ImportError('Failed to import NumPy')

    if sparseDef is False:
        sparseDef = '/users/modellers/pica/Data/proc/tides/sparse.def'

    tFile = tempfile.NamedTemporaryFile(delete=deleteFile)
    if noisy:
        if deleteFile is False:
            print 'Saving to temporary file {}...'.format(tFile.name)
        else:
            print 'Saving to temporary file...',


    np.savetxt(tFile.name, data, fmt='%4i/%02i/%02i %02i:%02i:%02i %.3f')

    if noisy:
        print 'done.'
        print 'Running TAPPy on the current station...',

    xFile = tempfile.NamedTemporaryFile()
    subprocess.call(['/usr/bin/tappy.py', 'analysis', '--def_filename=' + sparseDef, '--outputxml=' + xFile.name, '--quiet', tFile.name])

    [cName, cSpeed, cPhase, cAmplitude, cInference] = parseTAPPyXML(xFile.name)

    if noisy:
        print 'done.'

    return cName, cSpeed, cPhase, cAmplitude, cInference

def parseTAPPyXML(file):
    """
    Extract values from an XML file created by TAPPy.

    TODO: Allow a list of constituents to be specified when calling
    parseTAPPyXML.

    """

    from lxml import etree

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




