from __future__ import division

from collections import namedtuple
from datetime import datetime
from datetime import timedelta

import jdcal
import numpy as np
import pytz


from PyFVCOM.utilities.general import fix_range


def julian_day(gregorianDateTime, mjd=False):
    """
    For a given gregorian date format (YYYY,MM,DD,hh,mm,ss) get the
    Julian Day.

    Output array precision is the same as input precision, so if you
    want sub-day precision, make sure your input data are floats.

    Parameters
    ----------
    gregorianDateTime : ndarray
        Array of Gregorian dates formatted as [[YYYY, MM, DD, hh, mm,
        ss],...,[YYYY, MM, DD, hh, mm, ss]]. If hh, mm, ss are missing
        they are assumed to be zero (i.e. midnight).
    mjd : boolean, optional
        Set to True to convert output from Julian Day to Modified Julian
        Day.

    Returns
    -------
    jd : ndarray
        Modified Julian Day or Julian Day (depending on the value of
        mjd).

    Notes
    -----
    Julian Day epoch: 12:00 January 1, 4713 BC, Monday
    Modified Julain Day epoch: 00:00 November 17, 1858, Wednesday

    """

    try:
        nr, nc = np.shape(gregorianDateTime)
    except:
        nc = np.shape(gregorianDateTime)[0]
        nr = 1

    if nc < 6:
        # We're missing some aspect of the time. Let's assume it's the least
        # significant value (i.e. seconds first, then minutes, then hours).
        # Set missing values to zero.
        numMissing = 6 - nc
        if numMissing > 0:
            extraCols = np.zeros([nr, numMissing])
            if nr == 1:
                gregorianDateTime = np.hstack([gregorianDateTime, extraCols[0]])
            else:
                gregorianDateTime = np.hstack([gregorianDateTime, extraCols])

    if nr > 1:
        year = gregorianDateTime[:, 0]
        month = gregorianDateTime[:, 1]
        day = gregorianDateTime[:, 2]
        hour = gregorianDateTime[:, 3]
        minute = gregorianDateTime[:, 4]
        second = gregorianDateTime[:, 5]
    else:
        year = gregorianDateTime[0]
        month = gregorianDateTime[1]
        day = gregorianDateTime[2]
        hour = gregorianDateTime[3]
        minute = gregorianDateTime[4]
        second = gregorianDateTime[5]
    if nr == 1:
        julian, modified = jdcal.gcal2jd(year, month, day)
        modified += (hour + (minute / 60.0) + (second / 3600.0)) / 24.0
        julian += modified
    else:
        julian, modified = np.empty((nr, 1)), np.empty((nr, 1))
        for ii, tt in enumerate(gregorianDateTime):
            julian[ii], modified[ii] = jdcal.gcal2jd(tt[0], tt[1], tt[2])
            modified[ii] += (hour[ii] + (minute[ii] / 60.0) + (second[ii] / 3600.0)) / 24.0
            julian[ii] += modified[ii]

    if mjd:
        return modified
    else:
        return julian


def gregorian_date(julianDay, mjd=False):
    """
    For a given Julian Day convert to Gregorian date (YYYY, MM, DD, hh, mm,
    ss). Optionally convert from modified Julian Day with mjd=True).

    This function is adapted to Python from the MATLAB julian2greg.m function
    (http://www.mathworks.co.uk/matlabcentral/fileexchange/11410).

    Parameters
    ----------
    julianDay : ndarray
        Array of Julian Days
    mjd : boolean, optional
        Set to True if the input is Modified Julian Days.

    Returns
    -------
    greg : ndarray
        Array of [YYYY, MM, DD, hh, mm, ss].

    Example
    -------
    >>> greg = gregorian_date(np.array([53583.00390625, 55895.9765625]), mjd=True)
    >>> greg.astype(int)
    array([[2005,    8,    1,    0,    5,   37],
           [2011,   11,   30,   23,   26,   15])

    """

    if not mjd:
        # It's easier to use jdcal in Modified Julian Day
        julianDay -= 2400000.5

    try:
        nt = len(julianDay)
    except TypeError:
        nt = 1

    greg = np.empty((nt, 6))
    if nt == 1:
        ymdf = jdcal.jd2gcal(2400000.5, julianDay)
        fractionalday = ymdf[-1]
        hours = int(fractionalday * 24)
        minutes = int(((fractionalday * 24) - hours) * 60)
        seconds = ((((fractionalday * 24) - hours) * 60) - minutes) * 60
        greg = np.asarray((ymdf[0], ymdf[1], ymdf[2], hours, minutes, seconds))
    else:
        for ii, jj in enumerate(julianDay):
            ymdf = jdcal.jd2gcal(2400000.5, jj)
            greg[ii, :3] = ymdf[:3]
            fractionalday = ymdf[-1]
            hours = int(fractionalday * 24)
            minutes = int(((fractionalday * 24) - hours) * 60)
            seconds = ((((fractionalday * 24) - hours) * 60) - minutes) * 60
            greg[ii, 3:] = [hours, minutes, seconds]

    return greg


def date_range(start_date, end_date, inc=1):
    """
    Make a list of datetimes from start_date to end_date (inclusive).

    Parameters
    ----------
    start_date, end_date : datetime
        Start and end time as datetime objects. `end_date' is inclusive.
    inc : float, optional
        Specify a time increment for the list of dates in days. If omitted,
        defaults to 1 day.

    Returns
    -------
    dates : list
        List of datetimes.

    """

    start_seconds = int(start_date.replace(tzinfo=pytz.UTC).strftime('%s'))
    end_seconds = int(end_date.replace(tzinfo=pytz.UTC).strftime('%s'))

    inc *= 86400  # seconds
    dates = np.arange(start_seconds, end_seconds, inc)
    dates = [datetime.utcfromtimestamp(d) for d in dates]
    if dates[-1] != end_date:
        dates += [end_date]
    dates = np.array(dates)

    return dates


def overlap(t1start, t1end, t2start, t2end):
    """
    Find if two date ranges overlap.

    Parameters
    ----------
    datastart, dataend : datetime, float
        Observation start and end datetimes.
    modelstart, modelend : datetime, float
        Observation start and end datetimes.

    Returns
    -------
    overlap : bool
        True if the two periods overlap at all, False otherwise.

    References
    ----------
    Shamelessly copied from http://stackoverflow.com/questions/3721249

    """

    return (t1start <= t2start <= t1end) or (t2start <= t1start <= t2end)


def common_time(times1, times2):
    """
    Return the common date range in two time series. At least three dates are
    required for a valid overlapping time.

    Neither date range supplied need have the same sampling or number of
    times.

    If there is no overlap between the two time series, returns a tuple of
    (False, False).

    Parameters
    ----------
    times1 : list-like
        First time range (datetime objects). At least three values required.
    times2 : list-like
        Second time range (formatted as above).

    Returns
    -------
    common_time : tuple
        Start and end times indicating the common period between the two data
        sets.

    References
    ----------
    Shamelessly copied from https://stackoverflow.com/questions/9044084.

    """
    if len(times1) < 3 or len(times2) < 3:
        raise ValueError('Too few times for an overlap (times1 = {}, times2 = {})'.format(len(times1), len(times2)))

    # Check our times overlap at all and if not, return a couple of Falses.
    latest_start, earliest_end = False, False

    if overlap(times1[0], times1[-1], times2[0], times2[-1]):
        Range = namedtuple('Range', ['start', 'end'])
        r1 = Range(start=times1[0], end=times1[-1])
        r2 = Range(start=times2[0], end=times2[-1])
        latest_start = max(r1.start, r2.start)
        earliest_end = min(r1.end, r2.end)

    return latest_start, earliest_end


def make_signal(time, amplitude=1, phase=0, period=1):
    """
    Make an arbitrary sinusoidal signal with given amplitude, phase and period over a specific time interval.

    Parameters
    ----------
    time : np.ndarray
        Time series in number of days.
    amplitude : float, optional
        A specific amplitude (defaults to 1).
    phase : float, optional
        A given phase offset in degrees (defaults to 0).
    period : float, optional
        A period for the sine wave (defaults to 1).

    Returns
    -------
    signal : np.ndarray
        The time series with the given parameters.

    """

    signal = (amplitude * np.sin((2 * np.pi * 1 / period * (time - np.min(time)) + np.deg2rad(phase))))

    return signal


def ramped_signal(start, end, time_interval, signal_start, signal_end, min_rate, max_rate, ramp_duration):
    """
    Generate a constant time series which ramps over the given period (both upto and down from the rates).

    Parameters
    ----------
    start : str
        Start of the period for which to generate the signal, formatted '%Y-%m-%d %H:%M:%S'.
    end : str
        End date of periof for the signal. Same format as above.
    time_interval : float
        Increment in days for the time series.
    signal_start : str
        Leak start time, formatted '%Y-%m-%d %H:%M:%S'
    signal_end : str
        Leak end time, formatted '%Y-%m-%d %H:%M:%S'
    min_rate : float
        Minimum signal value.
    max_rate : float
        Maximum signal value.
    ramp_duration : list
        Periods in days over which to ramp up and down the signal from `min_rate' to `max_rate'. To only ramp up,
        specify the second element in the list as -1. If values are None, no ramping is applied.

    Returns
    -------
    t : np.ndarray
        Time series for the signal (datetime objects).
    signal : np.ndarray
        Leak values for the times in `t'.

    """

    if len(ramp_duration) > 2:
        raise UserWarning('Specify two ramps only (start and end).')

    if ramp_duration[0] is not None and ramp_duration[0] < time_interval:
        raise ValueError("The start ramp is shorter than the time interval. Specify either a bigger ramp or supply a "
                         "higher resolution time series (`t')")

    if ramp_duration[1] is not None and ramp_duration[1] < time_interval:
        raise ValueError("The end ramp is shorter than the time interval. Specify either a bigger ramp or supply a "
                         "higher resolution time series (`t')")

    start_datetime = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    run_duration = (end_datetime - start_datetime).days
    start_signal_datetime = datetime.strptime(signal_start, '%Y-%m-%d %H:%M:%S')
    end_signal_datetime = datetime.strptime(signal_end, '%Y-%m-%d %H:%M:%S')

    t = []
    for d in np.arange(run_duration):
        for h in np.arange(0, 1 - time_interval, time_interval):
            t.append(start_datetime + timedelta(days=float(d), hours=float(h) * 24.0))
    # Add the end of the requested period to the list and convert to array.
    t.append(end_datetime)
    t = np.asarray(t)

    signal = np.repeat(min_rate, len(t)).astype(float)

    if ramp_duration[0] is None:
        signal_start_index_offset = 0
    else:
        signal_start_index_offset = np.ceil(ramp_duration[0] / time_interval).astype(int)

    if ramp_duration[-1] is None:
        signal_end_index_offset = len(t) - 1
    elif ramp_duration[-1] < 0:
        signal_end_index_offset = signal_start_index_offset
    else:
        signal_end_index_offset = np.ceil(ramp_duration[-1] / time_interval).astype(int)

    start_index = np.argmin([np.abs((tt - start_signal_datetime).total_seconds()) for tt in t])
    start_max_index = start_index + signal_start_index_offset
    end_index = np.argmin([np.abs((tt - end_signal_datetime).total_seconds()) for tt in t])
    end_max_index = np.min((end_index + signal_end_index_offset, len(signal) - 1))

    hyperbolic_start = np.arange(-signal_start_index_offset / 2, signal_start_index_offset / 2)
    hyperbolic_end = np.arange(-signal_start_index_offset / 2, signal_end_index_offset / 2)
    ramp_start = np.tanh((hyperbolic_start / signal_start_index_offset) * np.pi * 2)
    ramp_end = np.tanh((hyperbolic_end / signal_end_index_offset) * np.pi * 2)
    if np.any(ramp_start):
        ramp_start = fix_range(ramp_start, min_rate, max_rate)
    if np.any(ramp_end):
        ramp_end = fix_range(ramp_end, min_rate, max_rate)

    signal[start_index:start_max_index] = ramp_start
    # If the end indices are the same, assume we're just running the signal to the end, otherwise taper it with the
    # reverse of the ramp up.
    if end_index != end_max_index:
        signal[start_max_index:end_index] = max_rate
        signal[end_index:end_max_index] = ramp_end[::-1]
    else:
        signal[start_max_index:end_index + 1] = max_rate

    return t, signal
