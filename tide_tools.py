"""
Aggregate the tidal database tools into a single file instead of having them
strewn all over the place.

"""

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

    For reference, to extract the M2 amplitude, phase and speed for Ilfracombe,
    the SQL statment would be:

    SELECT
        Amplitude.value,
        Phase.value,
        Speed.value
    FROM
        Amplitude join Phase join Speed
    WHERE
        Phase.constituentName is 'm2' and
        Speed.constituentName is 'm2' and
        Amplitude.constituentName is 'm2' and
        Phase.shortName is 'ILF' and
        Speed.shortName is 'ILF' and
        Amplitude.shortName is 'ILF';

    """

    try:
        import sqlite3
    except ImportError:
        sys.exit('Importing SQLite3 module failed')

    conn = sqlite3.connect(db)
    c = conn.cursor()


    # Create the necessary tables if they don't exist already
    c.execute('CREATE TABLE IF NOT EXISTS StationName (latDD FLOAT(10), lonDD FLOAT(10), shortName TEXT COLLATE nocase, longName TEXT COLLATE nocase)')
    c.execute('CREATE TABLE IF NOT EXISTS Amplitude (shortName TEXT COLLATE nocase, value FLOAT(10), constituentName TEXT COLLATE nocase, valueUnits TEXT COLLATE nocase, inferredConstituent TEXT COLLATE nocase)')
    c.execute('CREATE TABLE IF NOT EXISTS Phase (shortName TEXT COLLATE nocase, value FLOAT(10), constituentName TEXT COLLATE nocase, valueUnits TEXT COLLATE nocase, inferredConstituent TEXT COLLATE nocase)')
    c.execute('CREATE TABLE IF NOT EXISTS Speed (shortName TEXT COLLATE nocase, value FLOAT(10), constituentName TEXT COLLATE nocase, valueUnits TEXT COLLATE nocase, inferredConstituent TEXT COLLATE nocase)')

    if noisy:
        print 'amplitude, phase and speed.',
    for item in xrange(len(inferred)):
        c.execute('INSERT INTO Amplitude VALUES (?,?,?,?,?)',\
                  (stationName, amplitude[item], constituentName[item], 'metres', inferred[item]))
        c.execute('INSERT INTO Phase VALUES (?,?,?,?,?)',\
                  (stationName, phase[item], constituentName[item], 'degrees', inferred[item]))
        c.execute('INSERT INTO Speed VALUES (?,?,?,?,?)',\
                  (stationName, speed[item], constituentName[item], 'degrees per mean solar hour', inferred[item]))

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

    import sqlite3 as sql

    if noisy:
        print 'Getting data for %s from the database.' % table

    try:
        con = sql.connect(db)

        with con:
            cur = con.cursor()
            if startYear and endYear:
                # We've been given a range of data
                if startYear == endYear:
                    # We have the same start and end dates, so just do a
                    # simpler version
                    cur.execute("SELECT * FROM " + table + " WHERE " + \
                    table + ".year == " + str(startYear))
                else:
                    # We have a date range
                    cur.execute("SELECT * FROM " + table + " WHERE " + \
                    table + ".year > " + str(startYear) + \
                    " AND " + table + ".year < " + str(endYear))
            else:
                # Return all data
                cur.execute("SELECT * FROM " + table)
            # Now get the data in a format we might actually want to use
            data = cur.fetchall()

        con.close()

    except sql.Error, e:
        if con:
            con.close()
            print "Error %s:" % e.args[0]
            data = [False]

    return data
