import numpy as np
import sqlite3 as sq
import datetime as dt
import subprocess as sp
import glob as gb
import os

from PyFVCOM.grid import vincenty_distance
from PyFVCOM.read import FileReader
from PyFVCOM.plot import Time, Plotter

SQL_UNIX_EPOCH = dt.datetime(1970, 1, 1, 0, 0, 0)

"""
Validation a

This also includes the code to build the databases of 





"""


#########
# Generic code to build database
class validation_db():
    def __init__(self, db_name):
        if db_name[-3:] != '.db':
            db_name += '.db'
        self.conn = sq.connect(db_name)
        self.create_table_sql = {}
        self.retrieve_data_sql = {}
        self.c = self.conn.cursor()

    def execute_sql(self, sql_str):
        self.c.execute(sql_str)
        return self.c.fetchall()

    def make_create_table_sql(self, table_name, col_list):

        create_str = 'CREATE TABLE IF NOT EXISTS ' + table_name + ' ('
        for this_col in col_list:
            create_str += this_col
            create_str += ', '
        create_str = create_str[0:-2]
        create_str += ');'
        self.create_table_sql['create_' + table_name] = create_str

    def insert_into_table(self, table_name, data):
        no_rows = len(data)
        no_cols = len(data[0])
        qs_string = '('
        for this_x in range(no_cols):
            qs_string += '?,'
        qs_string = qs_string[:-1]
        qs_string += ')'

        if no_rows > 1:
            self.c.executemany('insert or ignore into ' + table_name + ' values ' + qs_string, data)
        elif no_rows == 1:
            self.c.execute('insert into ' + table_name + ' values ' + qs_string, data[0])
        self.conn.commit()

    def select_qry(self, table_name, where_str, select_str='*', order_by_str=None, inner_join_str=None, group_by_str=None):
        qry_string = 'select ' + select_str + ' from ' + table_name
        if inner_join_str:
            qry_string += ' inner join ' + inner_join_str
        if where_str:
            qry_string += ' where ' + where_str
        if order_by_str:
            qry_string += ' order by ' + order_by_str
        if group_by_str:
            qry_string += ' group by ' + group_by_str
        return self.execute_sql(qry_string)

    def close_conn(self):
        self.conn.close()


def dt_to_epochsec(time_to_convert):
    return (time_to_convert - SQL_UNIX_EPOCH).total_seconds()

def epochsec_to_dt(time_to_convert):
    return SQL_UNIX_EPOCH + dt.timedelta(seconds=time_to_convert)


#########
# Comparison against BODC tidal gauge dataset
"""


Example:





"""


def plot_tides(fvcom_file_str_list, db_name, plot_map=False, threshold=500):
    """
    Function description






    """

    # check list
    
    # check db_name

    tide_obs_database = db_tide(db_name)
    for this_str in hmm: 
        if ind == 0:
            fvcom_data = FileReader(this_file_str, variables=['h', 'zeta'])
        else:
            fvcom_data += FileReader(this_file_str, variables=['h', 'zeta'])

    tla_name, lon_lat = tide_obs_database.get_gauge_locations()
    comparison_dict = {}
    plot_dict = {}

    for this_name, this_loc in zip(tla_name, lon_lat):
        near_node = fvcom_data.closest_node(lon_lat, vincenty=True, threshold=threshold)
        if near_node:
            print('hmm')

    return plot_dict

def _make_normal_tide_series(h_series):
    height_series = h_series - np.mean(h_series)
    return height_series


"""
Code to define the database object and to construct the original database

"""

class db_tide(validation_db):
    def make_bodc_tables(self):
        # Make the complete set of empty tables for data to be inserted into (as defined in _add_sql_strings)
        # Insert information into the error flags table
        self._add_sql_strings()
        for this_table, this_str in self.create_table_sql.items():
            self.execute_sql(this_str)
        error_data = [(0, '', 'No error'), (1, 'M', 'Improbable value flagged by QC'),
                        (2, 'N', 'Null Value'), (3, 'T', 'Value interpolated from adjacent values')]
        self.insert_into_table('error_flags', error_data)

    def insert_tide_file(self, file_list):
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

    def get_gauge_locations(self):
        gauge_site_data = np.asarray(self.select_qry('sites', None))
        tla_name = gauge_site_data[:,1]
        lon_lat = np.asarray(gauge_site_data[:, 3:5], dtype=float)
        return tla_name, lon_lat

    def get_nearest_gauge_id(self, lat, lon):
        sites_lat_lon = np.asarray(self.select_qry('sites', None, 'site_id, lat, lon'))
        min_dist = 9999999999999
        closest_gauge_id = -999
        for this_row in sites_lat_lon:
            this_dist = vincenty_distance([lat, lon], [this_row[1], this_row[2]])
            if this_dist < min_dist:
                min_dist = this_dist
                closest_gauge_id = this_row[0]
        return int(closest_gauge_id), min_dist

    def _add_sql_strings(self):
        bodc_tables = {'gauge_obs': ['site_id integer NOT NULL', 'time_int integer NOT NULL',
                                     'elevation real NOT NULL', 'elevation_flag integer', 'residual real', 'residual_flag integer',
                                     'PRIMARY KEY (site_id, time_int)', 'FOREIGN KEY (site_id) REFERENCES sites(site_id)',
                                     'FOREIGN KEY (elevation_flag) REFERENCES error_flags(flag_id)',
                                     'FOREIGN KEY (residual_flag) REFERENCES error_flags(flag_id)'],
                       'sites': ['site_id integer NOT NULL', 'site_tla text NOT NULL', 'site_name text', 'lon real', 'lat real',
                                 'other_stuff text', 'PRIMARY KEY (site_id)'],
                       'error_flags': ['flag_id integer NOT NULL', 'flag_code text', 'flag_description text']}

        for this_key, this_val in bodc_tables.items():
            self.make_create_table_sql(this_key, this_val)


class bodc_annual_tide_file():
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
    def make_wco_tables(self):
        # Make the complete set of empty tables for data to be inserted into (as defined in _add_sql_strings)
        # Insert information into the error flags table
        self._add_sql_strings()
        for this_table, this_str in self.create_table_sql.items():
            self.execute_sql(this_str)
        sites_data = [(0, 'L4',-4.217,50.250, ' '), (1, 'E1',-4.368,50.035,' ')]
        self.insert_into_table('sites', sites_data)
        measurement_type_data = [(0,'CTD measurements'), (1, 'Surface buoy measurements')]
        self.insert_into_table('measurement_types', measurement_type_data)
        self.execute_sql('create index date_index on obs (time_int);')

    def _add_sql_strings(self):
        wco_tables = {'obs':['buoy_id integer NOT NULL', 'time_int integer NOT NULL',
                                    'depth real NOT NULL', 'temp real', 'salinity real', 'measurement_flag integer NOT NULL',
                                    'PRIMARY KEY (buoy_id, depth, measurement_flag, time_int)', 'FOREIGN KEY (buoy_id) REFERENCES sites(buoy_id)',
                                    'FOREIGN KEY (measurement_flag) REFERENCES measurement_types(measurement_flag)'],
                       'sites':['buoy_id integer NOT NULL', 'buoy_name text', 'lon real', 'lat real',
                                    'other_stuff text', 'PRIMARY KEY (buoy_id)'],
                       'measurement_types':['measurement_flag integer NOT NULL', 'measurement_description text', 'PRIMARY KEY (measurement_flag)']}

        for this_key, this_val in wco_tables.items():
            self.make_create_table_sql(this_key, this_val)

    def insert_CTD_file(self, filestr, buoy_id):
        file_obj = WCO_obs_file(filestr)
        self._insert_obs(file_obj, buoy_id, 0.0)

    def insert_buoy_file(self, filestr, buoy_id):
        file_obj = WCO_obs_file(filestr, depth=0)
        self._insert_obs(file_obj, buoy_id, 1.0)

    def insert_CTD_dir(self, dirstr, buoy_id):
        file_obj = CTD_dir(dirstr)
        self._insert_obs(file_obj, buoy_id, 0.0)

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

class WCO_obs_file():
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


"""
Do the comparison


"""

class comp_data():
    def __init__(self, buoy_list, file_list_or_probe_dir, wco_database, max_time_threshold=dt.timedelta(hours=1), max_depth_threshold = 100, probe_depths=None):
        self.file_list_or_probe_dir
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
                    model_all_dicts[this_buoy] = {'dt_time':mod_times, 'temp':mod_t_vals, 'salinity':mod_s_vals}

                first_file = False
            else:
                fvcom_data_reader = FileReader(this_file, ['temp', 'salinity'], dims={'node':[close_node]})
                for this_buoy in self.buoy_list:
                    model_all_dicts


        model_depths = fvcom_data_reader.grid.siglay * fvcom_data_reader.grid.h * -1
        for this_buoy in self.buoy_list:
            model_all_dicts[this_buoy]['depth'] = model_dp


        for this_buoy in self.buoy_list:
            self.model_data[this_buoy] = model_all_dicts[this_buoy]

    def model_closest_time(self, find_time):
        return closest_time

class comp_data_probe(comp_data):
    def retrieve_file_data():        
        for this_buoy in self.buoy_list:
            t_filelist = []
            s_filelist = []
            for this_dir in self.file_or_probe_dir_list:
                t_filelist.append(this_dir + this_buoy + '_t1.dat')
                s_filelist.append(this_dir + this_buoy + '_s1.dat')
            mod_times, mod_t_vals, mod_pos = pf.read.read_probes(t_filelist, locations=True, datetimes=True)
            mod_times, mod_s_vals, mod_pos = pf.read.read_probes(s_filelist, locations=True, datetimes=True)
            model_dict = {'dt_time':mod_times, 'temp':mod_t_vals, 'salinity':mod_s_vals}    
            self.model_data[this_buoy] = model_dict


def wco_model_comparison(model_file_list, obs_database_file):
    
    temp_comp = []
    sal_comp = []
    dates_comp = []
    max_obs_depth = []

    obs_dates = np.unique([this_date.date() for this_date in observations['time']])
    for this_ind, this_obs_date in enumerate(obs_dates):
        if this_obs_date >= model_time_mm[0].date() and this_obs_date <= model_time_mm[1].date():
            this_obs_choose = [this_time.date() == this_obs_date for this_time in observations['time']]
            this_obs_time = np.min(observations['time'][this_obs_choose])
            this_time_close = fvcom_data_reader.closest_time(this_obs_time)

            this_obs_deps = observations['h'][this_obs_choose]
            this_obs_temp = observations['temp'][this_obs_choose]
            this_obs_salinity = observations['salinity'][this_obs_choose]

            this_obs_temp_interp = np.squeeze(np.interp(model_depths, this_obs_deps, this_obs_temp))
            this_obs_salinity_interp = np.squeeze(np.interp(model_depths, this_obs_deps, this_obs_salinity))

            this_model_temp = np.squeeze(fvcom_data_reader.data.temp[this_time_close,...])
            this_model_salinity = np.squeeze(fvcom_data_reader.data.salinity[this_time_close,...])

            temp_comp.append(np.asarray([this_obs_temp_interp, this_model_temp]))
            sal_comp.append(np.asarray([this_obs_salinity_interp, this_model_salinity]))
            dates_comp.append(this_obs_time)
            max_obs_depth.append(np.max(this_obs_deps))
    
    ctd_comp = {'temp':temp_comp, 'salinity':sal_comp, 'dates':dates_comp, 'max_depth':max_obs_depth}
    
    observations = get_buoy_obs(obs_meta_data)
   
    if observations: 
        buoy_temp_comp = []
        buoy_sal_comp = []
        buoy_dates_comp = []

        for this_ind, this_obs_time in enumerate(observations['time']):
            if this_obs_time >= model_time_mm[0] and this_obs_time <= model_time_mm[1]:
                this_time_close = fvcom_data_reader.closest_time(this_obs_time)
                buoy_temp_comp.append([observations['temp'][this_ind], fvcom_data_reader.data.temp[this_time_close,0]])
                buoy_sal_comp.append([observations['salinity'][this_ind], fvcom_data_reader.data.salinity[this_time_close,0]])
                buoy_dates_comp.append([this_obs_time, this_time_close])

        buoy_comp = {'temp':buoy_temp_comp, 'salinity':buoy_sal_comp, 'dates':buoy_dates_comp}

    else:
        buoy_comp = {}

    return ctd_comp, buoy_comp


