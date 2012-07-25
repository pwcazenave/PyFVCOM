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
