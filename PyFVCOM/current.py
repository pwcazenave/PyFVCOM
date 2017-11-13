"""
Tools to work with current data. Reuses some functions from PyFVCOM.tide.

"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import copy

from datetime import datetime

from PyFVCOM.utilities import common_time
from PyFVCOM.grid import grid_metrics, shape_coefficients

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
        FVCOM : PyFVCOM.read.FileReader
            Times series object and grid. Must have loaded the current data (u,
            v).
        PREDICTED : PyFVCOM.read.FileReader
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
                # Get overlapping period between the two data sets.
                overlap = common_time(FVCOM.time.datetime, _new_time)
                # Find the indices for FVCOM.time.datetime for that overlap.
                f_start = _new_time.tolist().index(overlap[0])
                f_end = _new_time.tolist().index(overlap[1])
                # Check we've got a common time series now.
                if np.max(_new_time[f_start:f_end + 1] - FVCOM.time.datetime).total_seconds() == 0:
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
            if self._noisy:
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
                if self._noisy:
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


def scalar2vector(direction, magnitude):
    """
    Convert arrays of two scalars into the corresponding vector components.
    This is mainly meant to be used to convert direction and speed to the u and
    v velocity components.

    Parameters
    ----------
    direction, magnitude : ndarray
        Arrays of direction (degrees) and magnitude (any units).

    Returns
    -------
    u, v : ndarray
        Arrays of the u and v components of the magnitude and direction in units of
        magnitude.

    """

    u = np.sin(np.deg2rad(direction)) * magnitude
    v = np.cos(np.deg2rad(direction)) * magnitude

    return u, v


def vector2scalar(u, v):
    """
    Convert two vector components into the scalar values. Mainly used for
    converting u and v velocity components into direction and magnitude.

    Parameters
    ----------
    u, v : ndarray
        n-dimensional arrays of u and v vectors.

    Returns
    -------
    direction, magnitude : ndarray
        Arrays of direction (degrees) and magnitude (u and v units).

    """

    direction = np.rad2deg(np.arctan2(u, v))
    magnitude = np.hypot(u, v)

    return direction, magnitude


def residual_flow(FVCOM, idxRange=False, checkPlot=False, noisy=False):
    """
    Calculate the residual flow. By default, the calculation will take place
    over the entire duration of FVCOM['Times']. To limit the calculation to a
    specific range, give the index range as idxRange = [0, 100], for the first
    to 100th time step.  Alternatively, specify idxRange as 'daily' or
    'spring-neap' for daily and spring neap cycle residuals.

    Parameters
    ----------
    FVCOM : dict
        Contains the FVCOM model results.
    idxRange : list or str, optional
        If a list, the start and end index for the time series analysis.
        If a string, then must be one of 'daily' or 'spring-neap' to
        clip the time series to a day or a spring-neap cycle.
    checkPlot : int
        Plot a PVD at element checkPlot of the first vertical layer to
        check the code is working properly.
    noisy : bool
        Set to True to enable verbose output.

    Returns
    -------
    uRes : ndarray
        Raw summed velocity u-direction vector component. Useful for PVD
        plots.
    vRes : ndarray
        Raw summed velocity v-direction vector component. Useful for PVD
        plots.
    rDir : ndarray
        Residual direction array for each element centre in the
        unstructured grid.
    rMag : ndarray
        Residual magnitude array for each element centre in the
        unstructured grid.

    Notes
    -----
    Based on my MATLAB do_residual.m function.


    """

    toSecFactor = 24 * 60 * 60

    # Get the output interval (in days)
    dt = FVCOM['time'][2] - FVCOM['time'][1]

    # Some tidal assumptions. This will need to change in areas in which the
    # diurnal tide dominates over the semidiurnal.
    tideCycle = (12.0 + (25.0 / 60)) / 24.0
    # The number of values in the output file which covers a tidal cycle
    tideWindow = np.ceil(tideCycle / dt)

    # Get the number of output time steps which cover the selected period (in
    # idxRange). If it's spring-neap, use 14.4861 days; daily is one day,
    # obviously.

    startIdx = np.ceil(3 / dt)  # start at the third day to skip the warm up period

    if idxRange == 'spring-neap':
        # To the end of the spring-neap cycle
        endIdx = startIdx + tideWindow + np.ceil(14.4861 / dt)
    elif idxRange == 'daily':
        endIdx = startIdx + tideWindow + np.ceil(1 / dt)
    elif idxRange is False:
        startIdx = 0
        endIdx = -1
    else:
        startIdx = idxRange[0]
        endIdx = idxRange[1]

    try:
        # 3D input
        nTimeSteps, nLayers, nElements = np.shape(FVCOM['u'][startIdx:endIdx, :, :])
    except:
        # 2D input
        nTimeSteps, nElements = np.shape(FVCOM['u'][startIdx:endIdx, :])
        nLayers = 1

    tideDuration = ((dt * nTimeSteps) - tideCycle) * toSecFactor

    # Preallocate outputs.
    uRes = np.zeros([nTimeSteps, nLayers, nElements])
    vRes = np.zeros([nTimeSteps, nLayers, nElements])
    uSum = np.empty([nTimeSteps, nLayers, nElements])
    vSum = np.empty([nTimeSteps, nLayers, nElements])
    uStart = np.empty([nLayers, nElements])
    vStart = np.empty([nLayers, nElements])
    uEnd = np.empty([nLayers, nElements])
    vEnd = np.empty([nLayers, nElements])

    for hh in range(nLayers):
        if noisy:
            print('Layer {} of {}'.format(hh + 1, nLayers))

        try:
            # 3D
            uSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['u'][startIdx:endIdx, hh, :]), axis=0)
            vSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['v'][startIdx:endIdx, hh, :]), axis=0)
        except:
            # 2D
            uSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['u'][startIdx:endIdx, :]), axis=0)
            vSum[:, hh, :] = np.cumsum(np.squeeze(FVCOM['v'][startIdx:endIdx, :]), axis=0)

        for ii in range(nTimeSteps):
            # Create progressive vectors for all time steps in the current layer
            if noisy:
                if ii == 0 or np.mod(ii, 99) == 0:
                    print('Create PVD at time step {} of {}'.format(ii + 1, nTimeSteps))

            uRes[ii, hh, :] = uRes[ii, hh, :] + (uSum[ii, hh, :] * (dt * toSecFactor))
            vRes[ii, hh, :] = vRes[ii, hh, :] + (vSum[ii, hh, :] * (dt * toSecFactor))

        uStart[hh, :] = np.mean(np.squeeze(uRes[0:tideWindow, hh, :]), axis=0)
        vStart[hh, :] = np.mean(np.squeeze(vRes[0:tideWindow, hh, :]), axis=0)
        uEnd[hh, :] = np.mean(np.squeeze(uRes[-tideWindow:, hh, :]), axis=0)
        vEnd[hh, :] = np.mean(np.squeeze(vRes[-tideWindow:, hh, :]), axis=0)

    uDiff = uEnd - uStart
    vDiff = vEnd - vStart

    # Calculate direction and magnitude.
    rDir = np.arctan2(uDiff, vDiff) * (180 / np.pi)  # in degrees.
    rMag = np.sqrt(uDiff**2 + vDiff**2) / tideDuration  # in units/s.

    # Plot to check everything's OK
    if checkPlot:
        if noisy:
            print('Plotting element {}'.format(checkPlot - 1))

        elmt = checkPlot - 1
        lyr = 0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(uRes[:, lyr, elmt], vRes[:, lyr, elmt])
        ax.plot(uRes[0:tideWindow, lyr, elmt], vRes[0:tideWindow, lyr, elmt], 'gx')
        ax.plot(uRes[-tideWindow:, lyr, elmt], vRes[-tideWindow:, lyr, elmt], 'rx')
        ax.plot(uStart[lyr, elmt], vStart[lyr, elmt], 'go')
        ax.plot(uEnd[lyr, elmt], vEnd[lyr, elmt], 'ro')
        ax.plot([uStart[lyr, elmt], uEnd[lyr, elmt]], [vStart[lyr, elmt], vEnd[lyr, elmt]], 'k')
        ax.set_xlabel('Displacement west-east')
        ax.set_ylabel('Displacement north-south')
        ax.set_aspect('equal')
        ax.autoscale(tight=True)
        fig.show()

    return uRes, vRes, rDir, rMag


def vorticity(fvcom, depth_averaged=False):
    """
    This function computes the vorticity from some velocity data.

    Parameters
    ----------
    fvcom : PyFVCOM.FileReader
        FVCOM model output. Requires velocity data to be loaded (either depth-resolved ('u', 'v') or depth-averaged
        ('ua', 'va').
    depth_averaged : bool
        Set to True to compute the vorticity from the depth-averaged velocity data. Defaults to the depth-resolved data.

    Returns
    vort : np.ndarray
        Horizontal vorticity (1/s) array.

    Notes
    -----
        This can be slow over a large domain.
        This is lifted more or less directly from PySeidon (https://github.com/GrumpyNounours/PySeidon).

    """

    # Compute the elements connected to each element if we haven't already been given it. This is slow as it does a
    # bunch of other things too even though we only want element connectivity.
    if not hasattr(fvcom.grid, 'nbe'):
        _, _, fvcom.grid.nbe, isbce, _ = grid_metrics(fvcom.grid.triangles)

    # We need a1u and a2u to calculate vorticity. If we haven't got them, make them.
    if not hasattr(fvcom.grid, 'a1u') and not hasattr(fvcom.grid, 'a2u'):
        fvcom.grid.a1u, fvcom.grid.a2u = shape_coefficients(fvcom.grid.xc, fvcom.grid.yc, fvcom.grid.triangles, isbce)

    # Surrounding elements
    n1 = fvcom.grid.nbe[:, 0]
    n2 = fvcom.grid.nbe[:, 1]
    n3 = fvcom.grid.nbe[:, 2]

    if depth_averaged:
        if not hasattr(fvcom.data, 'ua') and not hasattr(fvcom.data, 'va'):
            raise AttributeError('Missing the depth-averaged velocity data.')
        dudy = np.zeros((fvcom.dims.time, fvcom.dims.nele))
        dvdx = np.zeros((fvcom.dims.time, fvcom.dims.nele))
        a1u = fvcom.grid.a1u
        a2u = fvcom.grid.a2u
        data_u = fvcom.data.ua
        data_v = fvcom.data.va
    else:
        if not hasattr(fvcom.data, 'u') and not hasattr(fvcom.data, 'v'):
            raise AttributeError('Missing the velocity data.')
        dudy = np.zeros((fvcom.dims.time, fvcom.dims.siglay, fvcom.dims.nele))
        dvdx = np.zeros((fvcom.dims.time, fvcom.dims.siglay, fvcom.dims.nele))
        a1u = fvcom.grid.a1u[:, np.newaxis, :]
        a2u = fvcom.grid.a2u[:, np.newaxis, :]
        data_u = fvcom.data.u
        data_v = fvcom.data.v

    for time_index in range(fvcom.dims.time):
        dvdx[time_index, ...] = (np.multiply(a1u[0, ...], data_v[time_index, ...])
                                 + np.multiply(a1u[1, ...].T, data_v[time_index, ..., n1]).T
                                 + np.multiply(a1u[2, ...].T, data_v[time_index, ..., n2]).T
                                 + np.multiply(a1u[3, ...].T, data_v[time_index, ..., n3]).T)
        dudy[time_index, ...] = (np.multiply(a2u[0, ...], data_u[time_index, ...])
                                 + np.multiply(a2u[1, ...].T, data_u[time_index, ..., n1]).T
                                 + np.multiply(a2u[2, ...].T, data_u[time_index, ..., n2]).T
                                 + np.multiply(a2u[3, ...].T, data_u[time_index, ..., n3]).T)

    vort = dvdx - dudy

    return vort


def ebb_flood(u, v, time, time_start=None, time_end=None):
    """
    Compute the flood and ebb windows for a given time series.

    Notes
    -----
    This is lifted from PySeidon and adjusted to work with our FileReader class.

    The definition of flood and ebb here is currents that are within +/- 90 degrees of the principal tidal axis.

    Parameters
    ----------
    u, v : np.ndarray
        Velocity component time series data ([time]).
    time : np.ndarray
        Array of datetime objects for the time series `u', `v'.
    time_start : str, datetime
        start time, as a string ('yyyy-mm-ddThh:mm:ss') or datetime object.
    time_end : str, datetime
        end time, as a string ('yyyy-mm-ddThh:mm:ss') or datetime object.

    Returns
    -------
    flood_indices : np.ndarray
        Indices indicating flood times.
    ebb_indices : np.ndarray, optional
        Indices indicating ebb times.
    principal_tidal_axis : np.ndarray, optional
        Principal flow axis (float number in degrees)
    principal_tidal_axis_variance : np.ndarray
        Associated variance (float number)

    Notes
    -----
    - May take time to compute if time period too long
    - Directions between -180 and 180 deg., i.e. 0=East, 90=North, +/-180=West, -90=South
    - Assumes that flood is aligned with principal direction

    """

    def closest_time(time, when):
        """ Find the index of the closest `time' to the supplied time `when' (datetime object). """
        return np.argmin(np.abs(time - when))

    nt = len(time)

    # Find time interval to work in.
    if isinstance(time_start, str):
        time_start = datetime.strftime(time_start, '%Y-%m-%dT%H:%M:%S')

    if isinstance(time_end, str):
        time_end= datetime.strftime(time_end, '%Y-%m-%dT%H:%M:%S')

    time_window = None
    if time_start and time_end:
        time_window = np.arange(closest_time(time, time_start), closest_time(time, time_end))
    elif time_start and not time_end:
        time_window = np.arange(closest_time(time, time_start), nt)
    elif not time_start and time_end:
        time_window = np.arange(0, closest_time(time, time_end))

    # Subset in time if requested to do so.
    if np.any(time_window):
        U = u[time_window]
        V = v[time_window]
    else:
        U = u[:]
        V = v[:]

    # WB version of BP's principal axis
    # Assuming principal axis = flood heading. Determine principal axes - potentially a problem if axes are very
    # kinked since this would misclassify part of ebb and flood.
    principal_tide_axis, principal_tide_axis_variance = principal_axis(U, V)

    # Put U and V the wrong way around in arctan2 to get north = 0 (otherwise east = 0).
    dir_all = np.rad2deg(np.arctan2(V, U))
    ind = np.where(dir_all < 0)
    dir_all[ind] = dir_all[ind] + 360
    # Sign speed - eliminating wrap-around
    dir_PA = dir_all - principal_tide_axis
    dir_PA[dir_PA < -90] += 360
    dir_PA[dir_PA > 270] -= 360
    # General direction of flood passed as input argument
    flood_indices = np.where((dir_PA >= -90) & (dir_PA < 90))
    ebb_indices = np.arange(dir_PA.shape[0])
    ebb_indices = np.delete(ebb_indices, flood_indices[:])
    # TR: quick fix
    if type(flood_indices).__name__ == 'tuple':
        flood_indices = flood_indices[0]

    return flood_indices, ebb_indices, principal_tide_axis, principal_tide_axis_variance


def principal_axis(u, v):
    """
    For a given pair of velocity components, compute the principal axis.

    Parameters
    ----------
    u, v : np.ndarray
        Velocity vectors ([x, time] or [x, depth, time])

    Returns
    -------
    principal_axis : float
        Principal axis in degrees.
    axis_variance : float
        Variance of the principal axis (in degrees).

    Notes
    -----
    This is slightly modified from PySeidon.utilities.BP_tools.

    """

    # Create velocity matrix
    U = np.vstack((u, v)).T
    # Eliminate NaN values
    U = U[~np.any(np.isnan(U), axis=1), ...]  # any time with a NaN in it
    # Convert matrix to deviate from
    rep = np.tile(np.mean(U, axis=0), [len(U), 1])
    U -= rep  # remove mean velocity so everything is centred about zero
    # Compute covariance matrix
    R = np.dot(U.T, U) / (len(U) - 1)

    # Calculate Eigenvalues and Eigenvectors for covariance matrix
    R[np.where(np.isnan(R))] = 0.0
    R[np.where(np.isinf(R))] = 0.0
    R[np.where(np.isneginf(R))] = 0.0
    lamb, V = np.linalg.eig(R)
    # Sort Eigenvalues in descending order so that major axis is given by first Eigenvector
    # Sort in descending order with indices
    ilamb = sorted(range(len(lamb)), key=lambda k: lamb[k], reverse=True)
    lamb = sorted(lamb, reverse=True)
    # Reconstruct the Eigenvalue matrix.
    lamb = np.diag(lamb)
    # Reorder the Eigenvectors.
    V = V[:, ilamb]

    # Rotation angle of major axis in radians relative to cartesian coordinates
    # ra = np.arctan2(V[0,1], V[1,1])
    ra = atan2(V[0,1], V[1,1])
    # Express principal axis in compass coordinates.
    # TR: still not sure here
    principal_axis = np.rad2deg(-ra)
    # Variance captured by principal axis.
    axis_variance = np.diag(lamb[0]) / np.trace(lamb)

    return principal_axis, axis_variance

