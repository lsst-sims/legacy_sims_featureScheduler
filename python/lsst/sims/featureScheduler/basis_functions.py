import numpy as np
import features
import utils
import healpy as hp


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
    Basis function to reward re-visiting an area on the sky. Looking for Solar System objects.
    """
    def __init__(self, survey_features=None, gap_min=15., gap_max=45., nside=32, npairs=1.):

        self.gap_min = gap_min/60./24.
        self.gap_max = gap_max/60./24.
        self.npairs = 1.

        if survey_features is None:
            self.survey_features = {}
            self.survey_features['Pair_in_night'] = features.Pair_in_night()
            self.survey_features['Last_observed'] = features.Last_observed()
            self.condition_features['Conditions'] = features.Conditions()
        super(Target_map_basis_function, self).__init__(survey_features=self.survey_features,
                                                        condition_features=self.condition_features)

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)
        diff = self.survey_features['Conditions']['mjd'] - self.survey_features['Last_observed'][indx]
        good = np.where((diff > self.gap_min) & (diff < self.gap_max) &
                        (self.survey_features['Pair_in_night'][indx] < self.npairs))
        result[indx][good] = 1.
        return result


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


