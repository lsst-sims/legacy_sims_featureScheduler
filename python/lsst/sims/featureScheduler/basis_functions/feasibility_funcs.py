import numpy as np
from lsst.sims.featureScheduler import features
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ['Filter_loaded_basis_function', 'Time_to_twilight_basis_function',
           'Not_twilight_basis_function', 'Force_delay_basis_function',
           'Hour_Angle_limit_basis_function', 'Moon_down_basis_function',
           'Fraction_of_obs_basis_function', 'Clouded_out_basis_function',
           'Rising_more_basis_function']


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
    alt_limit : int (18)
        The sun altitude limit to use. Must be 12 or 18
    """
    def __init__(self, time_needed=30., alt_limit=18):
        super(Time_to_twilight_basis_function, self).__init__()
        self.time_needed = time_needed/60./24.  # To days
        self.alt_limit = str(alt_limit)

    def check_feasibility(self, conditions):
        available_time = getattr(conditions, 'sun_n' + self.alt_limit + '_rising') - conditions.mjd
        result = available_time > self.time_needed
        return result


class Not_twilight_basis_function(Base_basis_function):
    def __init__(self, sun_alt_limit=-18):
        """
        # Should be -18 or -12
        """
        self.sun_alt_limit = str(sun_alt_limit).replace('-', 'n')
        super(Not_twilight_basis_function, self).__init__()

    def check_feasibility(self, conditions):
        result = True
        if conditions.mjd < getattr(conditions, 'sun_'+self.sun_alt_limit+'_setting'):
            result = False
        if conditions.mjd > getattr(conditions, 'sun_'+self.sun_alt_limit+'_rising'):
            result = False
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


class Look_ahead_ddf_basis_function(Base_basis_function):
    """Look into the future to decide if it's a good time to observe or block.

    Parameters
    ----------
    """
    def __init__(self, frac_total, RA=0., ha_limits=None, survey_name='', time_jump=44.):
        super(Look_ahead_ddf_basis_function, self).__init__()
        self.survey_name = survey_name
        self.frac_total = frac_total
        self.ra_hours = RA/360.*24.
        self.HA_limits = np.array(ha_limits)
        self.time_jump = time_jump
        self.survey_features['Ntot'] = features.N_obs_survey()
        self.survey_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_feasibility(self, conditions):
        result = True
        target_HA = (conditions.lmst - self.ra_hours) % 24
        # If it's more that self.time_jump to hour angle zero
        # See if there will be enough time to twilight in the future
        

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


class Rising_more_basis_function(Base_basis_function):
    """Say a spot is not available if it will rise substatially before twilight.

    Parameters
    ----------
    RA : float
        The RA of the point in the sky (degrees)
    pad : float
        When to start observations if there's plenty of time before twilight (minutes)
    """
    def __init__(self, RA, pad=30.):
        super(Rising_more_basis_function, self).__init__()
        self.RA_hours = RA * 24 / 360.
        self.pad = pad/60.  # To hours

    def check_feasibility(self, conditions):
        result = True
        hour_angle = conditions.lmst - self.RA_hours
        # If it's rising, and twilight is well beyond when it crosses the meridian
        time_to_twi = (conditions.sun_n18_rising - conditions.mjd)*24.
        if (hour_angle < -self.pad) & (np.abs(hour_angle) < (time_to_twi - self.pad)):
            result = False
        return result


## XXX--TODO:  Can include checks to see if downtime is coming, clouds are coming, moon rising, or surveys in a higher tier 
# Have observations they want to execute soon.