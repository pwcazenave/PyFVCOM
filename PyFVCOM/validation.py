import numpy as np
import sqlite3 as sq
import datetime as dt
import subprocess as sp
import glob as gb
import os
from pandas import read_hdf,DataFrame
import matplotlib.path as mplPath

import matplotlib.pyplot as plt

from PyFVCOM.grid import get_boundary_polygons, vincenty_distance
from PyFVCOM.read import FileReader
from PyFVCOM.plot import Time, Plotter
from PyFVCOM.stats import calculate_coefficient, rmse

SQL_UNIX_EPOCH = dt.datetime(1970, 1, 1, 0, 0, 0)

"""
Validation of model outputs against in situ data stored and extracted from a database.

This also includes the code to build the databases of time series data sets.

"""

class validation_db(object):
    """ Work with an SQLite database. """

    def __init__(self, db_name):
        """ Create a new database `db_name'.

        Parameters
        ----------
        db_name : str
            The path and name for the new database.

        """
        if db_name[-3:] != '.db':
            db_name += '.db'
        self.conn = sq.connect(db_name)
        self.create_table_sql = {}
        self.retrieve_data_sql = {}
        self.c = self.conn.cursor()

    def execute_sql(self, sql_str):
        """
        Execute the given SQL statement.

        Parameters
        ----------
        sql_str : str
            SQL statement to execute.

        Returns
        -------
        results : np.ndarray
            Data from the database which matches the SQL statement.
        """

        self.c.execute(sql_str)

        return self.c.fetchall()

    def create_table(self, table_name, col_list):
        """
        Create a database table if no such table exists.

        Parameters
        ----------
        table_name : str
            Table name to create.
        col_list : list
            List of column names to add to the table.

        """

        create_str = 'CREATE TABLE IF NOT EXISTS {table} ({cols});'.format(table=table_name, cols=', '.join(col_list))
        self.execute_sql(create_str)

    def insert_into_table(self, table_name, data, column=None):
        """
        Insert data into a table.

        Parameters
        ----------
        table_name : str
            Table name into which to insert the given data.
        data : np.ndarray
            Data to insert into the database.
        column : list, optional
            Insert the supplied `data' into this `column' within the given `table_name'.

        """

        data = np.asarray(data)
        if np.ndim(data) == 1:
            no_cols = len(data)
            no_rows = 1
            data = data[np.newaxis, :]
        else:
            no_rows, no_cols = data.shape
        qs_string = '({})'.format(','.join('?' * no_cols))

        # Format our optional column.
        if column is not None:
            column = '({})'.format(','.join(column))
        else:
            column = ''

        if no_rows > 1:
            self.c.executemany('insert or ignore into {tab} {col} values {val}'.format(tab=table_name, col=column, val=qs_string), data)
        elif no_rows == 1:
            self.c.execute('insert into {tab} {col} values {val}'.format(tab=table_name, col=column, val=qs_string), data[0])
        self.conn.commit()

    def select_qry(self, table_name, where_str, select_str='*', order_by_str=None, inner_join_str=None, group_by_str=None):
        """
        Extract data from the database which matches the given SQL query.

        Parameters
        ----------
        table_name : str
            Table name to query.
        where_str : str
            Where statement.
        select_str : str, optional
            Optionally give a set of columns to select.
        order_by_str : str, optional
            Optionally give a set of columns by which to order the results.
        inner_join_str : str, optional
            Optionally give an inner join string.
        group_by_str : str, optional
            Optionally give a string by which to group the results.

        Returns
        -------
        results : np.ndarray
            The data which matches the given query.

        """

        qry_string = 'select {} from {}'.format(select_str, table_name)
        if inner_join_str:
            qry_string += ' inner join {}'.format(inner_join_str)
        if where_str:
            qry_string += ' where {}'.format(where_str)
        if order_by_str:
            qry_string += ' order by {}'.format(order_by_str)
        if group_by_str:
            qry_string += ' group by {}'.format(group_by_str)

        return self.execute_sql(qry_string)

    def table_exists(self, variable):
        pass    
        #return table_exist    

    def close_conn(self):
        """ Close the connection to the database. """
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Tidy up the connection to the SQLite database. """
        self.conn.close()



def dt_to_epochsec(time_to_convert):
    """
    Convert a datetime to our SQL database epoch.

    Parameters
    ----------
    time_to_convert : datetime.datetime
        Datetime to convert.

    Returns
    -------
    epoched : int
        Converted datetime (in seconds).

    """

    return (time_to_convert - SQL_UNIX_EPOCH).total_seconds()


def epochsec_to_dt(time_to_convert):
    """

    Parameters
    ----------
    time_to_convert : int
        Seconds in the SQL database epoch.

    Return
    ------
    unepoched : datetime.datetime.
        Converted time.

    """
    return SQL_UNIX_EPOCH + dt.timedelta(seconds=time_to_convert)


def plot_map(fvcom, tide_db_path, threshold=np.inf, legend=False, **kwargs):
    """
    Plot the tide gauges which fall within the model domain (in space and time) defined by the given FileReader object.

    Parameters
    ----------
    fvcom : PyFVCOM.read.FileReader
        FVCOM model data as a FileReader object.
    tide_db_path : str
        Path to the tidal database.
    threshold : float, optional
        Give a threshold distance (in spherical units) beyond which a gauge is considered too far away.
    legend : bool, optional
        Set to True to add a legend to the plot. Defaults to False.

    Any remaining keyword arguments are passed to PyFVCOM.plot.Plotter.

    Returns
    -------
    plot : PyFVCOM.plot.Plotter
        The Plotter object instance for the map

    """

    tide_db = db_tide(tide_db_path)
    gauge_names, gauge_locations = tide_db.get_gauge_locations(long_names=True)

    gauges_in_domain = []
    fvcom_nodes = []
    for gi, gauge in enumerate(gauge_locations):
        river_index = fvcom.closest_node(gauge, threshold=threshold)
        if river_index:
            gauge_id, gauge_dist  = tide_db.get_nearest_gauge_id(*gauge)
            times, data = tide_db.get_tidal_series(gauge_id, np.min(fvcom.time.datetime), np.max(fvcom.time.datetime))
            if not np.any(data):
                continue

            gauges_in_domain.append(gi)
            fvcom_nodes.append(river_index)

    plot = Plotter(fvcom, **kwargs)
    fx, fy = plot.m(fvcom.grid.lon, fvcom.grid.lat)
    plot.plot_field(-fvcom.grid.h)
    plot.axes.plot(fx[fvcom_nodes], fy[fvcom_nodes], 'ro', markersize=3, zorder=202, label='Model')
    # Add the gauge locations.
    rx, ry = plot.m(gauge_locations[:, 0], gauge_locations[:, 1])
    plot.axes.plot(rx, ry, 'wo', label='Gauges')
    for xx, yy, name in zip(rx, ry, gauge_names[gauges_in_domain]):
        plot.axes.text(xx, yy, name, fontsize=10, rotation=45, rotation_mode='anchor', zorder=203)

    if legend:
        plot.axes.legend(numpoints=1, scatterpoints=1, ncol=2, loc='upper center', fontsize=10)

    return plot


def plot_tides(fvcom, db_name, threshold=500, figsize=(10, 10), **kwargs):
    """
    Plot model and tide gauge data.

    Parameters
    ----------
    fvcom : PyFVCOM.read.FileReader
        FVCOM model data as a FileReader object.
    db_name : str
        Database name to interrogate.
    threshold : float, optional
        Give a threshold distance (in spherical units) to exclude gauges too far from a model node.
    figsize : tuple
        Give a figure size (units are inches).

    Remaining keyword arguments are passed to PyFVCOM.plot.Time.

    Returns
    -------
    time : PyFVCOM.plot.Time
        Time series plot object.
    gauge_obs : dict
        Dictionary with the gauge and model data.

    """

    tide_db = db_tide(db_name)

    # Get all the gauges in the database and find the corresponding model nodes.
    gauge_names, gauge_locations = tide_db.get_gauge_locations(long_names=True)

    gauge_obs = {}
    gauges_in_domain = []
    fvcom_nodes = []
    for gi, gauge in enumerate(gauge_locations):
        river_index = fvcom.closest_node(gauge, threshold=threshold)
        if river_index:
            current_gauge = {}
            current_gauge['gauge_id'], current_gauge['gauge_dist'] = tide_db.get_nearest_gauge_id(*gauge)
            current_gauge['times'], current_gauge['data'] = tide_db.get_tidal_series(current_gauge['gauge_id'],
                                                                                     np.min(fvcom.time.datetime),
                                                                                     np.max(fvcom.time.datetime))
            if not np.any(current_gauge['data']):
                continue

            current_gauge['lon'], current_gauge['lat'] = gauge_locations[gi, :]

            current_gauge['gauge_clean'] = current_gauge['data'][:, 1] == 0
            current_gauge['gauge_obs_clean'] = {'times': np.copy(current_gauge['times'])[current_gauge['gauge_clean']],
                                                'data': np.copy(current_gauge['data'])[current_gauge['gauge_clean'], 0]}
            current_gauge['rescale_zeta'] = fvcom.data.zeta[:, river_index] - np.mean(fvcom.data.zeta[:, river_index])
            current_gauge['rescale_gauge_obs'] = current_gauge['gauge_obs_clean']['data'] - np.mean(current_gauge['gauge_obs_clean']['data'])

            current_gauge['dates_mod'] = np.isin(fvcom.time.datetime, current_gauge['gauge_obs_clean']['times'])
            current_gauge['dates_obs'] = np.isin(current_gauge['gauge_obs_clean']['times'], fvcom.time.datetime)
            # Skip out if we don't have any coincident data (might simply be a sampling issue) within the model
            # period. We should interpolate here.
            if not np.any(current_gauge['dates_mod']) or not np.any(current_gauge['dates_obs']):
                continue

            current_gauge['r'], current_gauge['p'] = calculate_coefficient(current_gauge['rescale_zeta'][current_gauge['dates_mod']], current_gauge['rescale_gauge_obs'][current_gauge['dates_obs']])
            current_gauge['rms'] = rmse(current_gauge['rescale_zeta'][current_gauge['dates_mod']], current_gauge['rescale_gauge_obs'][current_gauge['dates_obs']])
            current_gauge['std'] = np.std(current_gauge['rescale_zeta'][current_gauge['dates_mod']] - current_gauge['rescale_gauge_obs'][current_gauge['dates_obs']])

            gauges_in_domain.append(gi)
            fvcom_nodes.append(river_index)

            name = gauge_names[gi]
            gauge_obs[name] = current_gauge
            del current_gauge

    tide_db.close_conn()  # tidy up after ourselves

    # Now make a figure of all that data.
    if len(gauge_obs) > 5:
        cols = np.ceil(len(gauge_obs) ** (1.0 / 3)).astype(int) + 1
    else:
        cols = 1
    rows = np.ceil(len(gauge_obs) / cols).astype(int)
    fig = plt.figure(figsize=figsize)
    for count, site in enumerate(sorted(gauge_obs)):
        ax = fig.add_subplot(rows, cols, count + 1)
        time = Time(fvcom, figure=fig, axes=ax, hold=True, **kwargs)
        time.plot_line(gauge_obs[site]['rescale_zeta'], label='Model', color='k')
        # We have to use the raw plot function for the gauge data as the plot_line function assumes we're using model
        # data.
        time.axes.plot(gauge_obs[site]['gauge_obs_clean']['times'], gauge_obs[site]['rescale_gauge_obs'], label='Gauge', color='m')
        # Should add the times of the flagged data here.
        time.axes.set_xlim(fvcom.time.datetime.min(), fvcom.time.datetime.max())
        time.axes.set_ylim(np.min((gauge_obs[site]['rescale_gauge_obs'].min(), gauge_obs[site]['rescale_zeta'].min())),
                           np.max((gauge_obs[site]['rescale_gauge_obs'].max(), gauge_obs[site]['rescale_zeta'].max())))
        time.axes.set_title(site)

    return time, gauge_obs


def _make_normal_tide_series(h_series):
    height_series = h_series - np.mean(h_series)
    return height_series


class db_tide(validation_db):
    """ Create a time series database and query it. """

    def make_bodc_tables(self):
        """ Make the complete set of empty tables for data to be inserted into (as defined in _add_sql_strings) """

        # Insert information into the error flags table
        self._add_sql_strings()
        for this_key, this_val in self.bodc_tables.items():
            self.create_table(this_key, this_val)
        error_data = [(0, '', 'No error'), (1, 'M', 'Improbable value flagged by QC'),
                        (2, 'N', 'Null Value'), (3, 'T', 'Value interpolated from adjacent values')]
        self.insert_into_table('error_flags', error_data)

    def insert_tide_file(self, file_list):
        """
        Add data from a set of files to the database.

        Parameters
        ----------
        file_list : list
            List of file names.

        """
        for this_file in file_list:
            print('Inserting data from file: ' + this_file)
            this_file_obj = bodc_annual_tide_file(this_file)
            try:
                site_id = self.select_qry('sites', "site_tla == '" + this_file_obj.site_tla + "'", 'site_id')[0][0]
            except:
                try:
                    current_id_max = np.max(self.select_qry('sites', None, 'site_id'))
                    site_id = int(current_id_max + 1)
                except:
                    site_id = 1

                site_data = [(site_id, this_file_obj.site_tla, this_file_obj.site_name, this_file_obj.lon, this_file_obj.lat, '')]
                self.debug_data = site_data
                self.insert_into_table('sites', site_data)

            site_id_list = [site_id] * len(this_file_obj.seconds_from_ref)
            table_data = list(zip(site_id_list, this_file_obj.seconds_from_ref, this_file_obj.elevation_data,
                            this_file_obj.elevation_flag, this_file_obj.residual_data, this_file_obj.residual_flag))
            self.insert_into_table('gauge_obs', table_data)

    def get_tidal_series(self, station_identifier, start_date_dt=None, end_date_dt=None):
        """
        Extract a time series of tidal elevations for a given station.

        Parameters
        ----------
        station_identifier : str
            Database station identifier.
        start_date_dt, end_date_dt : datetime.datetime, optional
            Give start and/or end times to extract from the database. If omitted, all data are returned.

        Returns
        -------
        dates : np.ndarray
            Array of datetime objects.
        data : np.ndarray
            Surface elevation and residuals from the database for the given station.

        """
        select_str = "time_int, elevation, elevation_flag"
        table_name = "gauge_obs as go"
        inner_join_str = "sites as st on st.site_id = go.site_id"

        if isinstance(station_identifier, str):
            where_str = "st.site_tla = '" + station_identifier + "'"
        else:
            where_str = "st.site_id = " + str(int(station_identifier))

        if start_date_dt is not None:
            start_sec = dt_to_epochsec(start_date_dt)
            where_str += " and go.time_int >= " + str(start_sec)
        if end_date_dt is not None:
            end_sec = dt_to_epochsec(end_date_dt)
            where_str += " and go.time_int <= " + str(end_sec)
        order_by_str = 'go.time_int'
        return_data = self.select_qry(table_name, where_str, select_str, order_by_str, inner_join_str)
        if not return_data:
            print('No data available')
            dates, data = None, None
        else:
            return_data = np.asarray(return_data)
            date_list = [epochsec_to_dt(this_time) for this_time in return_data[:,0]]
            dates, data = np.asarray(date_list), return_data[:, 1:]

        return dates, data

    def get_gauge_locations(self, long_names=False):
        """
        Extract locations and names of the tide gauges from the database.

        Parameters
        ----------
        long_names : bool, optional
            If True, return the 'nice' long names rather than the station identifiers.

        Returns
        -------
        tla_name : np.ndarray
            List of tide gauge names.
        lon_lat : np.ndarray
            Positions of the gauges.

        """
        gauge_site_data = np.asarray(self.select_qry('sites', None))
        if long_names:
            tla_name = gauge_site_data[:, 2]
        else:
            tla_name = gauge_site_data[:, 1]
        lon_lat = np.asarray(gauge_site_data[:, 3:5], dtype=float)

        return tla_name, lon_lat

    def get_nearest_gauge_id(self, lon, lat, threshold=np.inf):
        """
        Get the ID of the gauge closest to the position given by `lon' and `lat'.

        lon, lat : float
            Position for which to search for the nearest tide gauge.

        Returns
        -------
        closest_gauge_id : int
            Database ID for the gauge closest to `lon' and `lat'.
        min_dist : float
            Distance in metres between `lon' and `lat' and the gauge.
        threshold : float
            Threshold distance in metres (inclusive) within which gauges must be from the given position. If no
            gauges are found within this distance, the gauge ID is None.

        """

        sites_lat_lon = np.asarray(self.select_qry('sites', None, 'site_id, lat, lon'))
        min_dist = np.inf
        closest_gauge_id = None  # we should make this False or None or something
        for this_row in sites_lat_lon:
            this_dist = vincenty_distance([lat, lon], [this_row[1], this_row[2]])
            if this_dist < min_dist:
                min_dist = this_dist
                closest_gauge_id = this_row[0]
        if min_dist >= threshold:
            closest_gauge_id = None
        else:
            closest_gauge_id = int(closest_gauge_id)

        return closest_gauge_id, min_dist

    def _add_sql_strings(self):
        """ Function to define the database structure. """
        self.bodc_tables = {'gauge_obs': ['site_id integer NOT NULL', 'time_int integer NOT NULL',
                                     'elevation real NOT NULL', 'elevation_flag integer', 'residual real', 'residual_flag integer',
                                     'PRIMARY KEY (site_id, time_int)', 'FOREIGN KEY (site_id) REFERENCES sites(site_id)',
                                     'FOREIGN KEY (elevation_flag) REFERENCES error_flags(flag_id)',
                                     'FOREIGN KEY (residual_flag) REFERENCES error_flags(flag_id)'],
                       'sites': ['site_id integer NOT NULL', 'site_tla text NOT NULL', 'site_name text', 'lon real', 'lat real',
                                 'other_stuff text', 'PRIMARY KEY (site_id)'],
                       'error_flags': ['flag_id integer NOT NULL', 'flag_code text', 'flag_description text']}


class bodc_annual_tide_file(object):
    def __init__(self, file_name, header_length=11):
        """
        Assumptions: file name of the form yearTLA.txt

        """
        bodc_annual_tide_file._clean_tide_file(file_name, header_length)
        with open(file_name) as f:
            header_lines = [next(f) for this_line in range(header_length)]

        for this_line in header_lines:
            if 'ongitude' in this_line:
                self.lon = [float(s) for s in this_line.split() if bodc_annual_tide_file._is_number(s)][0]
            if 'atitude' in this_line:
                self.lat = [float(s) for s in this_line.split() if bodc_annual_tide_file._is_number(s)][0]
            if 'Site' in this_line:
                site_str_raw = this_line.split()[1:]
                if len(site_str_raw) == 1:
                    site_str = site_str_raw[0]
                else:
                    site_str = ''
                    for this_str in site_str_raw:
                        site_str += this_str
        self.site_name = site_str
        self.site_tla = file_name.split('/')[-1][4:7]

        raw_data = np.loadtxt(file_name, skiprows=header_length, dtype=bytes).astype(str)

        seconds_from_ref = []
        for this_row in raw_data:
            this_dt_str = this_row[1] + ' ' + this_row[2]
            this_seconds_from_ref = dt_to_epochsec(dt.datetime.strptime(this_dt_str, '%Y/%m/%d %H:%M:%S'))
            seconds_from_ref.append(int(this_seconds_from_ref))
        self.seconds_from_ref = seconds_from_ref

        elevation_data = []
        elevation_flag = []
        residual_data = []
        residual_flag = []
        for this_row in raw_data:
            meas, error_code = bodc_annual_tide_file._parse_tide_obs(this_row[3])
            elevation_data.append(meas)
            elevation_flag.append(error_code)
            meas, error_code = bodc_annual_tide_file._parse_tide_obs(this_row[4])
            residual_data.append(meas)
            residual_flag.append(error_code)
        self.elevation_data = elevation_data
        self.elevation_flag = elevation_flag
        self.residual_data = residual_data
        self.residual_flag = residual_flag

    @staticmethod
    def _parse_tide_obs(in_str):
        error_code_dict = {'M':1, 'N':2, 'T':3}
        try:
            int(in_str[-1])
            error_code = 0
            meas = float(in_str)
        except:
            error_code_str = in_str[-1]
            meas = float(in_str[0:-1])
            try:
                error_code = error_code_dict[error_code_str]
            except:
                print('Unrecognised error code')
                return
        return meas, error_code

    @staticmethod
    def _is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    @staticmethod
    def _clean_tide_file(file_name, header_length):
        sed_str = "sed -i '"+ str(header_length + 1) + ",$ {/^ *[0-9]/!d}' " + file_name
        sp.call([sed_str], shell=True)


#################################################################################################################
"""


Validation against L4 and E1 CTD and buoy data

observations_meta_data = {'buoy_name':'E1', 'year':'2006', 'ctd_new_file_type': False,
            'ctd_datadir':'/data/euryale4/backup/mbe/Data/WCO_data/E1/CTD_data/2006',
            'buoy_filepath':None, 'lon':-4.368, 'lat':50.035}

observations_meta_data = {'buoy_name':'L4', 'year':'2015', 'ctd_new_file_type': True, 'ctd_filepath':'./data/e1_data_2015.txt',
            'buoy_filepath': , '/data/euryale4/backup/mbe/Data/WCO_data/L4/Buoy_data/l4_cont_data_2015.txt', 'lon':-4.217, 'lat':50.250}

model_filestr_lambda = lambda m: '/data/euryale4/backup/mbe/Models/FVCOM/tamar_v2/run/output/depth_tweak2/2006/{:02d}/tamar_v2_0001.nc'.format(m)
available_months = np.arange(1,13)
model_file_list = [model_filestr_lambda(this_month) for this_month in available_months]




"""

class db_wco(validation_db):
    """ Work with an SQL database of data from PML's Western Channel Observatory. """

    def make_wco_tables(self):
        """
        Make the complete set of empty tables for data to be inserted into (as defined in _add_sql_strings).

        """

        # Insert information into the error flags table
        self._add_sql_strings()
        for this_table, this_str in self.wco_tables.items():
            self.create_table(this_key, this_val)
        sites_data = [(0, 'L4',-4.217,50.250, ' '), (1, 'E1',-4.368,50.035,' ')]
        self.insert_into_table('sites', sites_data)
        measurement_type_data = [(0,'CTD measurements'), (1, 'Surface buoy measurements')]
        self.insert_into_table('measurement_types', measurement_type_data)
        self.execute_sql('create index date_index on obs (time_int);')

    def _add_sql_strings(self):
        self.wco_tables = {'sites':['buoy_id integer NOT NULL', 'buoy_name text', 'lon real', 'lat real',
                                    'other_stuff text', 'PRIMARY KEY (buoy_id)'],
                           'measurement_types':['measurement_flag integer NOT NULL', 'measurement_description text',
                                    'PRIMARY KEY (measurement_flag)']}
    
    def _add_new_variable_table(self, variable):
        this_table_sql = ['buoy_id integer NOT NULL', 'time_int integer NOT NULL',
                                    'depth real NOT NULL', variable + ' real', 'measurement_flag integer NOT NULL',
                                    'PRIMARY KEY (buoy_id, depth, measurement_flag, time_int)', 'FOREIGN KEY (buoy_id) REFERENCES sites(buoy_id)',
                                    'FOREIGN KEY (measurement_flag) REFERENCES measurement_types(measurement_flag)']
        self.create_table(variable, this_variable_table_sql)

    def insert_CTD_file(self, filestr, buoy_id):
        file_obj = WCO_obs_file(filestr)
        self._insert_obs(file_obj, buoy_id, 0.0)

    def insert_buoy_file(self, filestr, buoy_id):
        file_obj = WCO_obs_file(filestr, depth=0)
        self._insert_obs(file_obj, buoy_id, 1.0)

    def insert_CTD_dir(self, dirstr, buoy_id):
        file_obj = CTD_dir(dirstr)
        self._insert_obs(file_obj, buoy_id, 0.0)

    def insert_csv_file(self, filestr, buoy_id):
        file_obj = csv_formatted(filstr)
        self._insert_obs(file_obj, buoy_id)

    def _insert_obs(self, file_obj, buoy_id, measurement_id):
        epoch_sec_timelist = []
        for this_time in file_obj.observation_dict['dt_time']:
            epoch_sec_timelist.append(dt_to_epochsec(this_time))
        buoy_id_list = np.tile(buoy_id, len(epoch_sec_timelist))
        measurement_id_list = np.tile(measurement_id, len(epoch_sec_timelist))

        table_data = list(zip(buoy_id_list, epoch_sec_timelist, file_obj.observation_dict['depth'], file_obj.observation_dict['temp'],
                        file_obj.observation_dict['salinity'], measurement_id_list))
        self.insert_into_table('obs', table_data)

    def get_observations(self, buoy_name, start_date_dt=None, end_date_dt=None, measurement_id=None):
        select_str = "time_int, depth, temp, salinity"
        table_name = "obs as go"
        inner_join_str = "sites as st on st.buoy_id = go.buoy_id"

        where_str = "st.buoy_name = '" + buoy_name + "'"

        if start_date_dt is not None:
            start_sec = dt_to_epochsec(start_date_dt)
            where_str += " and go.time_int >= " + str(start_sec)
        if end_date_dt is not None:
            end_sec = dt_to_epochsec(end_date_dt)
            where_str += " and go.time_int <= " + str(end_sec)
        order_by_str = 'go.time_int, go.depth'

        return_data = self.select_qry(table_name, where_str, select_str, order_by_str, inner_join_str)

        if not return_data:
            dates, data = None, None
            print('No data available')
        else:
            return_data = np.asarray(return_data)
            date_list = [epochsec_to_dt(this_time) for this_time in return_data[:,0]]
            dates, data = np.asarray(date_list), return_data[:,1:]

        return dates, data


class WCO_obs_file(object):
    def __init__(self, filename, depth=None):
        self._setup_possible_vars()
        self.observation_dict = self._add_file(filename)
        if depth is not None:
            self.observation_dict['depth'] = np.tile(depth, len(self.observation_dict['dt_time']))

    def _add_file(self,filename,remove_undated=True):
        # remove duff lines
        sed_str = "sed '/^-9.990e/d' " + filename + " > temp_file.txt"
        sp.call(sed_str, shell=True)

        # some files have multiple records of differing types...helpful
        temp_str = 'YKA123ASD'
        file_split_str = '''awk '/^[^0-9]/{g++} { print $0 > "''' + temp_str + '''"g".txt"}' temp_file.txt'''
        sp.call(file_split_str, shell=True)
        temp_file_list = gb.glob(temp_str + '*')

        obs_dict_list = []
        for this_file in temp_file_list:
            this_obs = self._add_file_part(this_file)
            if not remove_undated or 'dt_time' in this_obs:
                obs_dict_list.append(this_obs)
        rm_file = [os.remove(this_file) for this_file in temp_file_list]
        return {this_key:np.hstack([this_dict[this_key] for this_dict in obs_dict_list]) for this_key in obs_dict_list[0]}

    def _add_file_part(self, filename):
        # seperate header and clean out non numeric lines
        head_str ="head -1 " + filename + " > temp_header_file.txt"
        sed_str = "sed '/^[!0-9]/!d' " + filename + " > temp_file.txt"
        sp.call(head_str, shell=True)
        sp.call(sed_str, shell=True)

        # Load the files, some use semi-colon delimiters, some whitespace...
        if ';' in str(np.loadtxt('temp_header_file.txt', delimiter='no_delimination_needed', dtype=str)):
            observations_raw = np.loadtxt('temp_file.txt', delimiter=';',dtype=str)
            observations_header = np.loadtxt('temp_header_file.txt', delimiter=';',dtype=str)
        elif ',' in str(np.loadtxt('temp_header_file.txt', delimiter='no_delimination_needed', dtype=str)):
            observations_raw = np.loadtxt('temp_file.txt', delimiter=',',dtype=str)
            observations_header = np.loadtxt('temp_header_file.txt', delimiter=',',dtype=str)
        else:
            observations_raw = np.loadtxt('temp_file.txt', dtype=str)
            observations_header = np.loadtxt('temp_header_file.txt', dtype=str)

        # Clean up temp files
        os.remove('temp_file.txt')
        os.remove('temp_header_file.txt')

        # Find the relevant columns and pull out temp, salinity, date, etc if available
        observation_dict = {}
        time_vars = []

        for this_var, this_possible in self.possible_vars.items():
            if np.any(np.isin(this_possible, observations_header)):
                this_col = np.where(np.isin(observations_header, this_possible))[0]
                if this_var == 'time' or this_var =='date' or this_var=='Jd':
                    observation_dict[this_var] = np.squeeze(np.asarray(observations_raw[:,this_col], dtype=str))
                    time_vars.append(this_possible[np.isin(this_possible, observations_header)])
                else:
                    observation_dict[this_var] = np.squeeze(np.asarray(observations_raw[:,this_col], dtype=float))
        if 'date' in observation_dict:
            observation_dict['dt_time'] = self._parse_dates_to_dt(observation_dict, time_vars)
        return observation_dict

    def _setup_possible_vars(self):
        self.possible_vars = {'temp':np.asarray(['Tv290C', 'SST', ' Mean SST (degC)']), 'salinity':np.asarray(['Sal00', 'Sal', ' Mean SST (degC)']),
                        'depth':np.asarray(['DepSM']),    'date':np.asarray(['mm/dd/yyyy', 'Year', ' Date (YYMMDD)']),
                        'julian_day':np.asarray(['Jd']), 'time':np.asarray(['hh:mm:ss', 'Time', ' Time (HHMMSS)'])}

    @staticmethod
    def _parse_dates_to_dt(obs_dict, time_vars):
        dt_list = []
        if np.any(np.isin('mm/dd/yyyy', time_vars)):
            for this_time, this_date in zip(obs_dict['time'], obs_dict['date']):
                dt_list.append(dt.datetime.strptime(this_date + ' ' + this_time, '%m/%d/%Y %H:%M:%S'))
        elif np.any(np.isin('Year', time_vars)):
            for this_time, (this_jd, this_year) in zip(obs_dict['time'], zip(obs_dict['julian_day'], obs_dict['date'])):
                dt_list.append(dt.datetime(int(this_year),1,1) + dt.timedelta(days=int(this_jd) -1) +
                                dt.timedelta(hours=int(this_time.split('.')[0])) + dt.timedelta(minutes=int(this_time.split('.')[1])))
        elif np.any(np.isin(' Date (YYMMDD)', time_vars)):
            for this_time, this_date in zip(obs_dict['time'], obs_dict['date']):
                dt_list.append(dt.datetime.strptime(this_date + ' ' + this_time, '%y%m%d %H%M%S'))
        else:
            print('Date parser not up to date with possible vars')
            dt_list = None

        return np.asarray(dt_list)


class CTD_dir(WCO_obs_file):
    def __init__(self, dirname):
        all_files = os.listdir(dirname)
        dt_list = []
        observation_dict_list = []
        self._setup_possible_vars()

        for this_file in all_files:
            print('Processing file {}'.format(this_file))
            try:
                observation_dict_list.append(self._add_file(dirname + this_file, remove_undated=False))
                date_str = '20' + this_file[0:2] + '-' + this_file[2:4] + '-' + this_file[4:6]
                this_dt = dt.datetime.strptime(date_str, '%Y-%m-%d') + dt.timedelta(hours=12)
                dt_list.append(np.tile(this_dt, len(observation_dict_list[-1]['temp'])))

            except ValueError:
                print('Error in file {}'.format(this_file))
        # Flatten the list of dictionaries to one dictionary
        self.observation_dict = {this_key:np.hstack([this_dict[this_key] for this_dict in observation_dict_list]) for this_key in observation_dict_list[0]}
        self.observation_dict['dt_time'] = np.hstack(dt_list)


class csv_formatted(object):
    def __init__(self, filename):
        pass

"""
Do the comparison


"""

class comp_data(object):
    def __init__(self, buoy_list, model_ident, wco_database, max_time_threshold=dt.timedelta(hours=1), max_depth_threshold = 100, probe_depths=None):
        self.model_ident = model_ident
        self.database_obj = wco_database
        self.buoy_list = buoy_list

        self.model_data = {}
        if probe_depths is not None:
            for this_ind, this_buoy in enumerate(buoy_list):
                self.model_data[this_buoy]['depth'] = probe_depths[this_ind]

        self.observations = {}
        for this_buoy in self.buoy_list:
            self.observations[this_buoy] = {}

    def retrieve_file_data(self):
        pass

    def retrieve_obs_data(self, buoy_name, var, measurement_type=None):
        if not hasattr(self.model_date_mm):
            print('Retrieve model data first')
            return
        obs_dt, obs_raw = self.database_obj.get_obs_data(buoy_name, var, self.model_date_mm[0], self.model_date_mm[1], measurement_type)
        obs_depth = obs_raw[:,0]
        obs_var = obs_raw[:,1]
        self.observations[buoy_name][var] = obs_dict

    def get_comp_data_interpolated(self, buoy_name, var_list):
        if not hasattr(self, comp_dict):
            self.comp_dict = {}

        obs_dates = np.unique([this_date.date() for this_date in observations['time']])
        obs_comp = {}
        for this_ind, this_obs_time in enumerate(obs_dates):
            if this_obs_date >= model_time_mm[0].date() and this_obs_date <= model_time_mm[1].date():
                this_obs_choose = [this_time.date() == this_obs_date for this_time in self.observations[buoy_name]['dt_time']]
                t
                this_time_before, this_time_after = self.model_closest_both_times(this_obs_time)

                this_obs_deps = self.observations[buoy_name]['depth'][this_obs_choose]
                for var in var_list:
                    this_obs = self.observations[buoy_name][var][this_obs_choose]
                    this_model = np.squeeze(fvcom_data_reader.data.temp[this_time_close,...])
                    this_model_interp = np.squeeze(np.interp(this_obs_deps, model_depths, this_model))

                    try:
                        obs_comp[var].append(this_comp)
                    except KeyError:
                        obs_comp[var] = this_comp

                    max_obs_depth.append(np.max(this_obs_deps))

        self.comp_dict[buoy_name] = obs_comp

    def comp_data_nearest(self, buoy_name, var_list):
        pass

    def model_closest_time():
        pass


class comp_data_filereader(comp_data):
    def retrieve_file_data(self):
        where_str = 'buoy_name in ('
        for this_buoy in self.buoy_list:
            where_str+=this_buoy + ','
        where_str = where_str[0:-1] + ')'
        buoy_lat_lons = self.wco_database.select_query('sites', where_str, 'buoy_name, lon, lat')

        first_file = True
        model_all_dicts = {}

        for this_file in self.file_list:
            if first_file:
                fvcom_data_reader = FileReader(this_file, ['temp', 'salinity'])
                close_node = []
                for this_buoy_ll in buoy_lat_lons:
                    close_node.append()

                model_depths = fvcom_data_reader.grid.siglay * fvcom_data_reader.grid.h * -1

                for this_buoy in self.buoy_list:
                    model_all_dicts[this_buoy] = {'dt_time': mod_times, 'temp': mod_t_vals, 'salinity': mod_s_vals}

                first_file = False
            else:
                fvcom_data_reader = FileReader(this_file, ['temp', 'salinity'], dims={'node':[close_node]})
                for this_buoy in self.buoy_list:
                    model_all_dicts

        model_depths = fvcom_data_reader.grid.siglay * fvcom_data_reader.grid.h * -1
        for this_buoy in self.buoy_list:
            model_all_dicts[this_buoy]['depth'] = model_dp
            self.model_data[this_buoy] = model_all_dicts[this_buoy]

    def model_closest_time(self, find_time):
        return closest_time


class comp_data_probe(comp_data):
    def retrieve_file_data():
        for this_buoy in self.buoy_list:
            t_filelist = []
            s_filelist = []
            for this_dir in self.model_ident:
                t_filelist.append(this_dir + this_buoy + '_t1.dat')
                s_filelist.append(this_dir + this_buoy + '_s1.dat')
            mod_times, mod_t_vals, mod_pos = pf.read.read_probes(t_filelist, locations=True, datetimes=True)
            mod_times, mod_s_vals, mod_pos = pf.read.read_probes(s_filelist, locations=True, datetimes=True)
            model_dict = {'dt_time':mod_times, 'temp':mod_t_vals, 'salinity':mod_s_vals}
            self.model_data[this_buoy] = model_dict


######################################################################################################################

""" 
Validation against ICES bottle data
"""

class ICES_comp(object):
    """
    A class for comparing FVCOM(-ERSEM) models to ICES bottle data. It is a fvcom-ised and class-ised version of code written 
    by Momme Butenschon for NEMO output.

    The ICES data used is in a premade h5 file. This how it was inherited and should be updated to a form we can reproduce. 
	Default ICES variables: 'TEMP','PSAL', 'DOXY(umol/l)', 'PHOS(umol/l)', 'SLCA(umol/l)', 'NTRA(umol/l)', 'AMON(umol/l)',
                            'PHPH', 'ALKY(mmol/l)', 'CPHL(mg/m^3)'

    Example
    -------
    from PyFVCOM.validation import ICES_comp
    import matplotlib.pyplot as plt

    datafile="/data/euryale4/backup/momm/Data/ICES-data/CTD-bottle/EX187716.averaged.sorted.reindexed.h5"
    modelroot="/data/euryale2/scratch/mbe/Models_2/FVCOM/rosa/output"
    years=[2005]
    months = [2,3]
    modelfile=lambda y,m: "{}/{}/{:02d}/aqua_v16_0001.nc".format(modelroot,y,m)
    modelfilelist = [modelfile(years[0], this_month) for this_month in months]

    test_comp = ICES_comp(modelfilelist, datafile, noisy=True)
    temp_ices, temp_model = test_comp.get_var_comp('TEMP')
    plt.scatter(temp_model, temp_ices)
    plt.xlabel('Modelled temperature')
    plt.ylabel('Observed temperature')


    To Do
    -----
    Make script to generate the ICES datafile
    Parrellelelise
    Add plotting
    Change null value to non numeric


    """

    def __init__(self, modelfilelist, ices_hdf_file, var_list=None, daily_avg=False, noisy=False):
        """
        Retrieves the ICES data from the timeperiod of the model run and from within its bounding polygon, then for
        each observation retrieves the nearest model output in space and time. These data are held in dicts self.ices_data
        and self.model_data, with keys corresponding to the ICES variable names.


        Parameters
        ----------
        modelfilelist : list-like
            List of strings of the locations of . It is assumed the files are in sequential time order.
        ices_hdf_file : string
            The path to the hdf file of ICES data.
        var_list : list-like, optional
            The variables for comparison (ICES names). If not specified then defaults to all available (see self._add_default_varlist)
        daily_avg : boolean, optional
            Set to true if comparing daily averaged FVCOM output
        noisy : boolean, optional
            Output progress strings

        """
        self.model_files = modelfilelist
        self.ices_file = ices_hdf_file
        self.daily_avg = daily_avg
        self._add_ICES_model_varnames()
        self._add_data_dicts()
        self.noisy = noisy        

        if var_list:
            self.var_keys = var_list
        else:
            self._add_default_varlist()

        model_varkeys = []
        for this_key in self.var_keys:
            this_model_key = self.ices_model_conversion[this_key] 
            if "+" in this_model_key:
                vlist = this_model_key.split('+')
                for v in vlist:
                    model_varkeys.append(v)
            else:
                model_varkeys.append(this_model_key)                   
        self.model_varkeys = model_varkeys

        self.zeta_filereader = FileReader(self.model_files[0], ['zeta'])
        if len(self.model_files) > 1:
            for this_file in self.model_files[1:]:
                self.zeta_filereader += FileReader(this_file, ['zeta'])
        self.lon_mm = [np.min(self.zeta_filereader.grid.lon), np.max(self.zeta_filereader.grid.lon)]
        self.lat_mm = [np.min(self.zeta_filereader.grid.lat), np.max(self.zeta_filereader.grid.lat)]
        bn_list = get_boundary_polygons(self.zeta_filereader.grid.triangles)[0] # Assumes first poly is outer boundary
        self.bnd_poly = mplPath.Path(np.asarray([self.zeta_filereader.grid.lon[bn_list], self.zeta_filereader.grid.lat[bn_list]]).T)
        self._ICES_dataget()
        self._model_dataget()

    def get_var_comp(self, var, return_locs_depths_dates = False):
        """
        Retreive the comparison data for a single variable

        Parameters
        ----------
        var : string
            The variable to return, this must be the same as something in the self.var_keys (i.e. the ICES name)
        return_locs_depths_dates : boolean, optional
            As well as data return the locations, depths, and dates of the ICES observations to allow subsetting of results


        Returns
        -------
        ices_data : array or dict
            array of observations or (if return_locs_depths_dates) a dict with observations, depths, dates, and locations
        model_data : array
            array of nearest model data corresponding to observations
        """

        if var not in self.var_keys:
            print('Variable not in retrieved data')
            return None, None

        ices_data = np.asarray(self.ices_data[var])
        model_data = np.asarray(self.model_data[var])
    
        remove_data = model_data < -100
        ices_data = ices_data[~remove_data]
        model_data = model_data[~remove_data]

        if return_locs_depths_dates:
            ices_dates = np.asarray(self.ices_data['time_dt'])[~remove_data]    
            ices_depths = np.asarray(self.ices_data['z'])[~remove_data]
            ices_lon = np.asarray(self.ices_data['lon'])[~remove_data]
            ices_lat = np.asarray(self.ices_data['lat'])[~remove_data]

            ices_data = {var:ices_data, 'depths':ices_depths, 'dates':ices_dates, 'lon':ices_lon, 'lat':ices_lat}    
        
        return ices_data, model_data

    def _ICES_dataget(self):
        # Read the ices datafile
        df=read_hdf(self.ices_file,"df")

        start_step_len = 1000000
        end_step_len = 10
        start_index = 0

        # The dataframe is huge so skip through to the approriate start point
        while start_step_len >= end_step_len:
            start_index = self._year_start(np.min(self.zeta_filereader.time.datetime).year, start_index, start_step_len, df)
            start_step_len = start_step_len/10
        df = df[int(start_index):]

        for n,sample in df.iterrows():
            if self.noisy:
                print('ICES sample {}'.format(n))

            h = int(np.floor(sample['Hr']/100))
            sample_dt = dt.datetime(int(sample['Year']),int(sample['Mnth']),int(sample['Dy']),h,int(sample['Hr'] - h*100))
            if sample_dt > np.max(self.zeta_filereader.time.datetime):
                break
            
            if self.lon_mm[0]<=sample['Longdeg']<=self.lon_mm[1] and self.lat_mm[0]<=sample['Latdeg']<=self.lat_mm[1] and \
                                    sample_dt >= np.min(self.zeta_filereader.time.datetime):

                if self.bnd_poly.contains_point(np.asarray([sample['Longdeg'], sample['Latdeg']])):
                    node_ind = self.zeta_filereader.closest_node([sample['Longdeg'], sample['Latdeg']], haversine=True)

                    if self.daily_avg: # For daily averages match by day, otherwise use nearest time
                        sample_dt = dt.datetime(int(sample['Year']),int(sample['Mnth']),int(sample['Dy']))

                    model_time_ind = self.zeta_filereader.closest_time(sample_dt)
                    model_dt = self.zeta_filereader.time.datetime[model_time_ind]

                    sample_depth=sample['d/p']
                    this_depth = self.zeta_filereader.grid.h[node_ind] + self.zeta_filereader.data.zeta[model_time_ind, node_ind]
                    dep_layers = this_depth * -1 * self.zeta_filereader.grid.siglay[:, node_ind]
                    z_ind = self._checkDepth(sample_depth, dep_layers)

                    if z_ind>=0 and self._checkSample(sample):
                        self._addICESsample(sample, sample_dt, model_dt, node_ind, z_ind)

    def _addICESsample(self, sample, sample_dt, model_dt, node_ind, z_ind):
        self.ices_data['time_dt'].append(sample_dt)    
        self.model_data['time_dt'].append(model_dt)
                        
        self.ices_data['lat'].append(sample['Latdeg'])
        self.ices_data['lon'].append(sample['Longdeg'])
        self.ices_data['z'].append(sample["d/p"])
        self.model_data['node_ind'].append(node_ind)
        self.model_data['z_ind'].append(z_ind)
    
        for key in self.var_keys:
            if sample[key]<-8.999999:
                mvalue=-1.e15
                dvalue=-1.e15
            else:
                dvalue=sample[key]
                mvalue=-1.e10
            self.ices_data[key].append(dvalue)
            self.model_data[key].append(mvalue)
        
        if 'NTRA(umol/l)' in self.var_keys:
            if sample['NTRA(umol/l)']>-8.999999 and sample['NTRI(umol/l)']>-8.999999:
                dvalue=sample['NTRI(umol/l)']
                mvalue=-1.e10
            else:
                dvalue=-1.e15
                mvalue=-1.e15
            self.ices_data['NTRI(umol/l)'].append(dvalue)
            self.model_data['NTRI(umol/l)'].append(mvalue)    

    def _model_dataget(self):
        if len(self.ices_data['time_dt']) == 0:
            print('No ICES data loaded for comparison')
            return
        
        current_modelfile_ind = 0
        current_modelfile_dt = [this_date.date() for this_date in FileReader(self.model_files[current_modelfile_ind]).time.datetime]
        unique_obs_days = np.unique([this_date.date() for this_date in self.ices_data['time_dt']])
        for counter_ind, this_day in enumerate(unique_obs_days):
            if self.noisy:
                print('Getting model data from day {} of {}'.format(counter_ind +1, len(unique_obs_days)))

            if this_day > current_modelfile_dt[-1]:
                current_modelfile_ind += 1
                if current_modelfile_ind < len(self.model_files):
                    current_modelfile_dt = [this_date.date() for this_date in FileReader(self.model_files[current_modelfile_ind]).time.datetime]
                else:
                    return
            this_day_index = np.where(np.asarray(current_modelfile_dt) == this_day)[0]
            this_day_fr = FileReader(self.model_files[current_modelfile_ind], self.model_varkeys, 
                                                                    dims={'time':np.arange(np.min(this_day_index),np.max(this_day_index)+1)}) 
            this_day_obs_inds = np.where(np.asarray([this_dt.date() for this_dt in self.ices_data['time_dt']]) == this_day)[0]
    
            for this_record_ind in this_day_obs_inds:
                for key in self.var_keys:
                    if self.ices_data[key][this_record_ind] >-9.99e9:
                        this_model_key = self.ices_model_conversion[key]
                        space_ind = self.model_data['node_ind'][this_record_ind]
                        dep_ind = self.model_data['z_ind'][this_record_ind] 
                        time_ind = this_day_fr.closest_time(self.ices_data['time_dt'][this_record_ind])

                        if "+" in this_model_key:
                            vlist = this_model_key.split('+')
                            mbuffer = 0
                            for v in vlist:
                                mbuffer += getattr(this_day_fr.data, v)[time_ind, dep_ind, space_ind]
                            self.model_data[key][this_record_ind] = mbuffer
                        else:
                            self.model_data[key][this_record_ind] = getattr(this_day_fr.data, this_model_key)[time_ind, dep_ind, space_ind]

    def _add_ICES_model_varnames(self):
        self.ices_model_conversion = {'TEMP':'temp', 'PSAL':'salinity', 'PHOS(umol/l)':'N1_p', 'SLCA(umol/l)':'N5_s',
                        'PHPH':'O3_pH', 'ALKY(mmol/l)':'O3_TA', 'NTRA(umol/l)':'N3_n', 'AMON(umol/l)':'N4_n',
                        'DOXY(umol/l)':'O2_o', 'CPHL(mg/m^3)':'P1_Chl+P2_Chl+P3_Chl+P4_Chl'}

    def _add_data_dicts(self):
        self.ices_data = {'lat':[], 'lon':[], 'z':[], 'time_dt':[], 'NTRI(umol/l)':[]}
        self.model_data = {'node_ind':[], 'z_ind':[], 'time_dt':[], 'NTRI(umol/l)':[]}
        for this_key in self.ices_model_conversion:
            self.ices_data[this_key] = []
            self.model_data[this_key] = []

    def _add_default_varlist(self):
        self.var_keys = ['TEMP','PSAL', 'DOXY(umol/l)', 'PHOS(umol/l)', 'SLCA(umol/l)', 'NTRA(umol/l)', 'AMON(umol/l)',
                            'PHPH', 'ALKY(mmol/l)', 'CPHL(mg/m^3)']

    def _checkSample(self, sample):
        hasData=False
        for key in self.var_keys:
            if sample[key]>-8.999999:
                hasData=True
        return hasData

    @staticmethod
    def _checkDepth(z,dep_lays_choose):
        if z>dep_lays_choose[-1]:
            return -1
        else:
            k=((z-dep_lays_choose)**2).argmin()
            return k

    @staticmethod
    def _year_start(year_find, start_index, step, df):
        year_found = 0
        this_ind = start_index
        while year_found == 0:
            this_year = df.loc[this_ind]['Year']
            if this_year >=year_find:
                year_found = 1
                return_step = this_ind - step
            this_ind = this_ind + step
        return int(return_step)
