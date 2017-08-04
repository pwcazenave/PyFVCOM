"""
Tools to work with current data. Reuses some functions from tide_tools.

"""

import numpy as np
import copy

class Residuals:

    def __init__(self, FVCOM, PREDICTED, periods=None, max_speed=None, noisy=False):
        """
        Calculate the residuals (differences between two time series).
        Optionally average over some period.

        This was originally written to allow me to compare model output with
        predicted time series (i.e.  reconstructed from a harmonic analysis of
        that model output).

        Parameters
        ----------
        FVCOM : PyFVCOM.read_results.FileReader
            Times series object and grid. Must have loaded the current data (u,
            v).
        PREDICTED : PyFVCOM.read_results.FileReader
            Times series object which is subtracted from `FVCOM'.
        periods : list-like
            Give a list of strings for averaging the residuals. Choose from
            `daily' or `monthly'. The latter assumes a month is 30 days.
        max_speed : dict
            Dictionary of values at which to cap speeds (and the velocity
            components). This makes for neater quiver plots. Valid key names
            are `instantaneous', `daily' and `monthly'). Units are the same as
            the velocity data.
        noisy : bool
            Set to True to enable verbose output (defaults to False).

        """

        self._noisy = noisy
        self._fvcom = FVCOM
        self._pred = PREDICTED
        self._max_speed = max_speed

        # Make the differences in the currents.
        if noisy:
            print('Compute residuals')

        # Check we've got comparable sampling. Interpolate or subsample if not.
        _inc_fvcom = np.unique(np.diff(FVCOM.time.datetime))[0].seconds / 60 / 60  # in hours
        _inc_predicted = np.unique(np.diff(PREDICTED.time.datetime))[0].seconds / 60 / 60  # in hours
        if _inc_fvcom != _inc_predicted:
            # Try subsampling first as that should yield the fastest result.
            if _inc_fvcom > _inc_predicted:
                skip = int(_inc_fvcom / _inc_predicted)
                _new_time = PREDICTED.time.datetime[::skip]
                if np.max(_new_time - FVCOM.time.datetime).total_seconds() == 0:
                    # Do the subsampling on the rest of the data.
                    for var in PREDICTED.obj_iter(PREDICTED.data):
                        setattr(PREDICTED.data, var, getattr(PREDICTED.data, var)[::skip, ...])
                    for time in PREDICTED.obj_iter(PREDICTED.time):
                        setattr(PREDICTED.time, time, getattr(PREDICTED.time, time)[::skip])
                else:
                    # We need to interpolate.
                    print('On my todo list...')
                    pass
            elif _inc_fvcom < _inc_predicted:
                skip = int(_inc_predicted / _inc_fvcom)
                _new_time = FVCOM.time.datetime[::skip]
                if np.max(_new_time - PREDICTED.time.datetime).total_seconds() == 0:
                    # Do the subsampling on the rest of the data.
                    for var in FVCOM.obj_iter(FVCOM.data):
                        setattr(FVCOM.data, var, getattr(FVCOM.data, var)[::skip, ...])
                    for time in FVCOM.obj_iter(FVCOM.time):
                        setattr(FVCOM.time, time, getattr(FVCOM.time, time)[::skip])
                else:
                    # We need to interpolate.
                    print('On my todo list...')
                    pass

        # Now make a common time variable for use throughout (PREDICTED and
        # FVCOM should be identical in length now).
        self.time = copy.copy(FVCOM.time)
        self._inc = np.unique(np.diff(self.time.datetime))[0].total_seconds() / 60 / 60  # in hours

        self.u_diff = self._fvcom.data.u - self._pred.data.u
        self.v_diff = self._fvcom.data.v - self._pred.data.v
        self.direction, self.speed = vector2scalar(self.u_diff, self.v_diff)

        # Cap the speed vectors at some maximum value for the quiver plots.
        if self._max_speed and 'instantaneous' in self._max_speed:
            if noisy:
                print('Capping maximum residual speed to {}m/s'.format(self._max_speed['instantaneous']))
            self.speed = self._clip(self.speed, self._max_speed['instantaneous'])

        # Make the components after we've clipped so our plots look nice.
        self.u_res = np.sin(np.deg2rad(self.direction)) * self.speed
        self.v_res = np.cos(np.deg2rad(self.direction)) * self.speed

        if periods:
            for period in periods:
                self.average(period)

    def _clip(self, field, limit):
        """ Clip a `field' to a given `limit'. """

        return np.where(field < limit, field, limit)

    def average(self, period):
        """
        Average the residuals over a given period.

        Parameters
        ----------
        period : str
            Choose 'daily' or 'monthly' for the averaging period.

        Adds a new object (`daily' or `monthly') with `speed', `direction',
        `u_res' and `v_res' arrays to match the main Residual object.

        """

        # Get the necessary bits of time.
        daily_inc = int(24.0 / self._inc)  # get model steps per day
        monthly_inc = daily_inc * 30  # get model steps per 30 days (~month)
        nt = self.time.datetime.shape[0]
        dnt = self.time.datetime[::daily_inc].shape[0]
        # Monthly increment might mean we end up trying to skip over more days
        # than we have in the input. So, set the number of montly times to be
        # 1 (only one month of data) or the number of times when subsampled at
        # the monthly increment, whichever is larger.
        mnt = np.max((self.time.datetime[::monthly_inc].shape[0], 1))
        nx = self.u_diff.shape[-1]

        # Prepare the arrays if we're doing averaging
        if period == 'daily':
            self.daily = type('daily', (object,), {})()
            if self._noisy:
                print('Compute daily residuals')
            for var in ('u_diff', 'v_diff'):
                datetime = []  # has to be a list so we can insert a datetime object.
                daily = np.empty((dnt, nx))
                # This could be done with a neat reshaping, but I can't be
                # bothered to figure it out, so we'll just do it the
                # old-fashioned way.
                for ti, t in enumerate(np.arange(0, nt, daily_inc).astype(int)):
                    daily[ti, :] = np.median(getattr(self, var)[t:t + daily_inc, :], axis=0)
                    datetime.append(self.time.datetime[np.min((t, nt - 1))])
                if 'instantaneous' in self._max_speed:
                    daily = self._clip(daily, self._max_speed['instantaneous'])
                setattr(self.daily, var, daily)
                setattr(self.daily, 'datetime', np.asarray(datetime))

            # Now create the speed and direction arrays.
            setattr(self.daily, 'speed', np.sqrt(getattr(self.daily, 'u_diff')**2 + getattr(self.daily, 'v_diff')**2))
            setattr(self.daily, 'direction', np.rad2deg(np.arctan2(getattr(self.daily, 'u_diff'), getattr(self.daily, 'v_diff'))))

            # Make the components after we've clipped so our plots look nice.
            self.daily.u_res = np.sin(np.deg2rad(self.daily.direction)) * self.daily.speed
            self.daily.v_res = np.cos(np.deg2rad(self.daily.direction)) * self.daily.speed

        elif period == 'monthly':
            self.monthly = type('monthly', (object,), {})()
            if self._noisy:
                print('Compute monthly residuals')
            for var in ('u_diff', 'v_diff'):
                datetime = []  # has to be a list so we can insert a datetime object.
                monthly = np.empty((mnt, nx))
                # This could be done with a neat reshaping, but I can't be
                # bothered to figure it out, so we'll just do it the
                # old-fashioned way.
                for ti, t in enumerate(np.arange(0, nt / 60 / 30, monthly_inc).astype(int)):
                    monthly[ti, :] = np.median(getattr(self, var)[t:t + monthly_inc, :], axis=0)
                    datetime.append(self.time.datetime[np.min(((t + monthly_inc) // 2, nt - 1))])  # mid-point
                setattr(self.monthly, var, monthly)
                setattr(self.monthly, 'datetime', np.asarray(datetime))

            # Now create the speed and direction arrays.
            setattr(self.monthly, 'speed', np.sqrt(getattr(self.monthly, 'u_diff')**2 + getattr(self.monthly, 'v_diff')**2))
            setattr(self.monthly, 'direction', np.rad2deg(np.arctan2(getattr(self.monthly, 'u_diff'), getattr(self.monthly, 'v_diff'))))

            if 'monthly' in self._max_speed:
                if noisy:
                    print('Capping monthly residuals to {} m/s'.format(self._max_speed['monthly']))
                self.monthly.speed = self._clip(self.monthly.speed, self._max_speed['monthly'])
            # Make the components after we've clipped so our plots look nice.
            self.monthly.u_res = np.sin(np.deg2rad(self.monthly.direction)) * self.monthly.speed
            self.monthly.v_res = np.cos(np.deg2rad(self.monthly.direction)) * self.monthly.speed

        if period == 'monthly':
            # We need to add a pseudo-time dimension to the monthly data so we
            # can still use the plot_var function.
            if np.ndim(self.monthly.speed) == 1:
                self.monthly.speed = self.monthly.speed[np.newaxis, :]
                self.monthly.direction = self.monthly.direction[np.newaxis, :]
                self.monthly.u_res = self.monthly.u_res[np.newaxis, :]
                self.monthly.v_res = self.monthly.v_res[np.newaxis, :]


def scalar2vector(direction, speed):
    """
    Convert arrays of two scalars into the corresponding vector components.
    This is mainly meant to be used to convert direction and speed to the u and
    v velocity components.

    Parameters
    ----------
    direction, speed : ndarray
        Arrays of direction (degrees) and speed (any units).

    Returns
    -------
    u, v : ndarray
        Arrays of the u and v components of the speed and direction in units of
        speed.

    """

    u = np.sin(np.deg2rad(direction)) * speed
    v = np.cos(np.deg2rad(direction)) * speed

    return u, v


def vector2scalar(u, v):
    """
    Convert two vector components into the scalar values. Mainly used for
    converting u and v velocity components into direction and speed.

    Parameters
    ----------
    u, v : ndarray
        Arrays of (optionally time, space (vertical and horizontal) varying)
        u and v vectors.

    Returns
    -------
    direction, speed : ndarray
        Arrays of direction (degrees) and speed (u and v units).

    """

    direction = np.rad2deg(np.arctan2(u, v))
    speed = np.sqrt(u**2 + v**2)

    return direction, speed
