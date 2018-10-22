import numpy as np
import numpy.ma as ma
from lsst.sims.featureScheduler import features
from lsst.sims.featureScheduler import utils
import healpy as hp
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ['Filter_loaded_basis_function', 'Time_to_twilight_basis_function',
           'Not_twilight_basis_function']


class Filter_loaded_basis_function(Base_basis_function):
    def __init__(self, filternames='r'):
        """
        Parameters
        ----------
        filternames : str of list of str
            The filternames that need to be mounted to execute.
        """

        super(Filter_loaded_basis_function, self).__init__()
        if type(filternames) is not list:
            filternames = [filternames]
        self.filternames = filternames

    def check_feasibility(self, conditions):

        for filtername in self.filternames:
            result = filtername in conditions.mounted_filters
            if result is False:
                return result
        return result


class Time_to_twilight_basis_function(Base_basis_function):
    def __init__(self, time_needed=30.):
        """
        Parameters
        ----------
        time_needed : float (30.)
            The time needed to run a survey (mintues).
        """
        super(Time_to_twilight_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days

    def check_feasibility(self, conditions):
        available_time = conditions.next_twilight_start - conditions.mjd
        result = available_time > self.time_needed
        return result


class Not_twilight_basis_function(Base_basis_function):
    def __init__(self, sun_alt_limit=-18.):
        """
        """
        self.sun_alt_limit = np.radians(sun_alt_limit)
        super(Not_twilight_basis_function, self).__init__()

    def check_feasibility(self, conditions):

        result = conditions.sunAlt < self.sun_alt_limit
        return result


class Force_delay_basis_function(Base_basis_function):
    def __init__(self, days_delay=2., survey_name=None):
        """
        Parameters
        ----------
        days_delay : float (2)
            The number of days to force a gap on.
        """
        super(Not_twilight_basis_function, self).__init__()
        self.days_delay = days_delay
        self.survey_name = survey_name
        self.survey_features['last_obs_self'] = features.Last_observation(survey_name=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd - self.extra_features['last_obs_self'].feature['mjd'] < self.day_space:
            result = False
        return result


class Hour_Angle_limit_(Base_basis_function):
    def __init__(self, RA=0., ha_limits=None):
        """
        Parameters
        ----------

        """
        super(Not_twilight_basis_function, self).__init__()
        self.ra_hours = RA/360.*24.
        self.HA_limits = np.array(ha_limits)

    def check_feasibility(self, conditions):
        target_HA = (conditions.lmst - self.ra_hours) % 24
        # Are we in any of the possible windows
        result = False
        for limit in self.HA_limits:
            lres = limit[0] <= target_HA < limit[1]
            result = result or lres

        return result


class Moon_down_basis_function(Base_basis_function):
    def check_feasibility(self, conditions):
        result = True
        if conditions.moonAlt > 0:
            result = False
        return result


## XXX--TODO:  Can include checks to see if downtime is coming, clouds are coming, moon rising, or surveys in a higher tier 
# Have observations they want to execute soon.