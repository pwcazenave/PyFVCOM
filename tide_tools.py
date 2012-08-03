"""
Aggregate the tidal database tools into a single file instead of having them
strewn all over the place.

"""

try:
    import sqlite3
except ImportError:
    sys.exit('Importing SQLite3 module failed')

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




