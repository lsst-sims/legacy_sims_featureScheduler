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
            result = self.filtername in conditions.mounted_filters
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

## XXX--TODO:  Can include checks to see if downtime is coming, clouds are coming, moon rising, or surveys in a higher tier 
# Have observations they want to execute soon.