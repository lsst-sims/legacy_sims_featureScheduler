import numpy as np
from utils import empty_observation


class BaseSurvey(object):
    def __init__(self, basis_functions, extra_features=None):
        """
        Parameters
        ----------
        basis_functions : list
            List of basis_function objects
        extra_features : list
            List of any additional features the survey may want to use
            e.g., for computing final dither positions, or feasability maps.
        """

        # XXX-Check that input is a list of features
        self.basis_functions = basis_functions
        self.cost = None
        if extra_features is None:
            self.extra_features = []
        else:
            self.extra_features = extra_features

    def add_observation(self, observation, **kwargs):
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for feature in self.extra_features:
            feature.add_observation(observation, **kwargs)

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            feature.update_conditions(conditions, **kwargs)

    def _check_feasability(self):
        """
        Check if the survey is feasable in the current conditions
        """
        return True

    def calc_cost_function(self):
        if self._check_feasability():
            self.cost = 0
            for bf in self.basis_functions:
                cost += bf.cost()
                if np.isinf(np.max(cost)):
                    return np.inf
            return cost
        else:
            return np.inf

    def return_observations(self):
        pass


class Deep_drill_survey(BaseSurvey):
    """
    Class to make deep drilling fields
    """
    def __init__(self, basis_functions, extra_features=None, sequence=None,
                 exptime=30., RA=0, dec=0):
        """
        Parameters
        ----------
        sequence : list
            Should be a list of strings specifying which filters to take, e.g.,
            ['r', 'r', 'i', 'i', 'z', 'y']
        """
        super(Deep_drill_survey, self).__init__(basis_functions=basis_functions, extra_features=None)
        self.sequence = sequence
        self.exptime = exptime
        self.RA = RA
        self.dec = dec

    def return_observations(self):
        result = []
        for fn in self.sequence:
            obs = empty_observation()
            # XXX--Note that we'll want to put some dithering schemes in here eventually. 
            obs['RA'] = self.RA
            obs['Dec'] = self.dec
            obs['exptime'] = self.exptime
            obs['filter'] = fn
            result.append(obs)
        return result


class Raster_survey(BaseSurvey):

