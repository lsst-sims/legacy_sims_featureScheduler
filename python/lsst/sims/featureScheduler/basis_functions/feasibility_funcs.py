import numpy as np
from lsst.sims.featureScheduler import features
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ['Filter_loaded_basis_function', 'Time_to_twilight_basis_function',
           'Not_twilight_basis_function', 'Force_delay_basis_function',
           'Hour_Angle_limit_basis_function', 'Moon_down_basis_function',
           'Fraction_of_obs_basis_function', 'Clouded_out_basis_function']


class Filter_loaded_basis_function(Base_basis_function):
    """Check that the filter(s) needed are loaded

    Parameters
    ----------
    filternames : str or list of str
        The filternames that need to be mounted to execute.
    """
    def __init__(self, filternames='r'):
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
    """Make sure there is enough time before twilight. Useful
    if you want to check before starting a long sequence of observations.

    Parameters
    ----------
    time_needed : float (30.)
        The time needed to run a survey (mintues).
    """
    def __init__(self, time_needed=30.):
        super(Time_to_twilight_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days

    def check_feasibility(self, conditions):
        available_time = conditions.next_twilight_start - conditions.mjd
        result = available_time > self.time_needed
        return result


class Not_twilight_basis_function(Base_basis_function):
    """Test that it is not currrently twilight time

    Parameters
    ----------
    sun_alt_limit : float (-18.)
        The altitude of the sun to consider the start of twilight (degrees)
    """
    def __init__(self, sun_alt_limit=-18.):
        """
        """
        self.sun_alt_limit = np.radians(sun_alt_limit)
        super(Not_twilight_basis_function, self).__init__()

    def check_feasibility(self, conditions):
        # XXX--TODO:  Update to use the info on the night, since expected
        # sunalt can be slightly off due to interpolation.
        result = conditions.sunAlt < self.sun_alt_limit
        return result


class Force_delay_basis_function(Base_basis_function):
    """Keep a survey from executing to rapidly.

    Parameters
    ----------
    days_delay : float (2)
        The number of days to force a gap on.
    """
    def __init__(self, days_delay=2., survey_name=None):
        super(Force_delay_basis_function, self).__init__()
        self.days_delay = days_delay
        self.survey_name = survey_name
        self.survey_features['last_obs_self'] = features.Last_observation(survey_name=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd - self.survey_features['last_obs_self'].feature['mjd'] < self.days_delay:
            result = False
        return result


class Hour_Angle_limit_basis_function(Base_basis_function):
    """Only execute a survey in limited hour angle ranges. Useful for
    limiting Deep Drilling Fields.

    Parameters
    ----------
    RA : float (0.)
        RA of the target (degrees).
    ha_limits : list of lists
        limits for what hour angles are acceptable (hours). e.g.,
        to give 4 hour window around RA=0, ha_limits=[[22,24], [0,2]]
    """
    def __init__(self, RA=0., ha_limits=None):
        super(Hour_Angle_limit_basis_function, self).__init__()
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
    """Demand the moon is down """
    def check_feasibility(self, conditions):
        result = True
        if conditions.moonAlt > 0:
            result = False
        return result


class Fraction_of_obs_basis_function(Base_basis_function):
    """Limit the fraction of all observations that can be labled a certain
    survey name. Useful for keeping DDFs from exceeding a given fraction of the
    total survey.

    Parameters
    ----------
    frac_total : float
        The fraction of total observations that can be of this survey
    survey_name : str
        The name of the survey
    """
    def __init__(self, frac_total, survey_name=''):
        super(Fraction_of_obs_basis_function, self).__init__()
        self.survey_name = survey_name
        self.frac_total = frac_total
        self.survey_features['Ntot'] = features.N_obs_survey()
        self.survey_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_feasibility(self, conditions):
        # If nothing has been observed, fine to go
        result = True
        if self.survey_features['Ntot'].feature == 0:
            return result
        ratio = self.survey_features['N_survey'].feature / self.survey_features['Ntot'].feature
        if ratio > self.frac_total:
            result = False
        return result


class Clouded_out_basis_function(Base_basis_function):
    def __init__(self, cloud_limit=0.7):
        super(Clouded_out_basis_function, self).__init__()
        self.cloud_limit = cloud_limit

    def check_feasibility(self, conditions):
        result = True
        if conditions.bulk_cloud > self.cloud_limit:
            result = False
        return result

## XXX--TODO:  Can include checks to see if downtime is coming, clouds are coming, moon rising, or surveys in a higher tier 
# Have observations they want to execute soon.