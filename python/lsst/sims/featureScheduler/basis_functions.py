import numpy as np
import features
import utils
import healpy as hp
from lsst.sims.sky_brightness_pre import M5percentiles


class Base_basis_function(object):
    """
    Class that takes features and computes a reward fucntion
    """

    def __init__(self, survey_features=None, condition_features=None, **kwargs):
        """

        """
        if survey_features is None:
            self.survey_features = {}
        else:
            self.survey_features = survey_features
        if condition_features is None:
            self.condition_features = {}
        else:
            self.condition_features = condition_features

    def add_observation(self, observation):
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation)

    def update_conditions(self, conditions):
        for feature in self.condition_features:
            self.conditions[feature].update_conditions(conditions)

    def __call__(self, **kwargs):
        """
        Return a reward healpix map or a reward scalar.
        """
        pass


class Target_map_basis_function(Base_basis_function):
    """
    Generate a map that rewards survey areas falling behind.
    """
    def __init__(self, filtername='r', nside=32, target_map=None, softening=1., survey_features=None,
                 condition_features=None):
        """
        Parameters
        ----------

        """
        if survey_features is None:
            self.survey_features = {}
            self.survey_features['N_obs'] = features.N_observations(fltername=filtername)
            self.survey_features['N_obs_reference'] = features.N_obs_reference()
        super(Target_map_basis_function, self).__init__(survey_features=self.survey_features,
                                                        condition_features=condition_features)
        self.nside = nside
        self.softening = softening
        if target_map is None:
            self.target_map = utils.generate_goal_map(filtername=filtername)
        else:
            self.target_map = target_map

    def __call__(self, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed

        Returns
        -------
        
        """
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)
        if indx is None:
            indx = np.arange(result.size)
        result[indx] = -self.survey_features['N_obs'][indx]
        result[indx] /= (self.survey_features['N_obs_reference'][indx] + self.softening)
        result[indx] += self.target_map[indx]
        return result


class Visit_repeat_basis_function(Base_basis_function):
    """
    Basis function to reward re-visiting an area on the sky.
    """
    def __init__(self):
        pass


class Depth_percentile_basis_function(Base_basis_function):
    """
    Return a healpix map of the reward function based on 5-sigma limiting depth
    """
    def __init__(self, survey_features=None, condition_features=None, filtername='r', nside=32):
        self.filtername = filtername
        self.nside = nside
        # Need conditions of sky brightness in filter, seeing_map, airmass_map.

    def __call__(self, indx=None):

        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)
        if indx is None:
            indx = np.arange(result.size)


