import numpy as np
from utils import empty_observation, set_default_nside
from lsst.sims.utils import _hpid2RaDec

default_nside = set_default_nside()


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
        self.reward = None
        if extra_features is None:
            self.extra_features = []
        else:
            self.extra_features = extra_features
        self.reward_checked = False

    def add_observation(self, observation, **kwargs):
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for feature in self.extra_features:
            feature.add_observation(observation, **kwargs)
        self.reward_checked = False

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            feature.update_conditions(conditions, **kwargs)
        self.reward_checked = False

    def _check_feasability(self):
        """
        Check if the survey is feasable in the current conditions
        """
        return True

    def calc_reward_function(self):
        self.reward_checked = True
        if self._check_feasability():
            self.reward = 0
            for bf in self.basis_functions:
                self.reward += bf.reward()
                if np.isinf(np.max(self.reward)):
                    self.reward = np.inf
        else:
            self.reward = np.inf
        return self.reward

    def return_observations(self):
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            reward = self.calc_reward_function()
        obs = empty_observation()
        return obs


class Simple_greedy_survey(BaseSurvey):
    """
    Just point at the healpixel with the heighest reward.
    XXX-NOTE THIS IS A BAD IDEA!
    XXX-Healpixels are NOT "evenly distributed" on the sky. Using them as pointing centers
    will result in features in the coadded depth power spectrum (I think).
    """
    def __init__(self, basis_functions, extra_features=None, filtername='r'):
        super(Simple_greedy_survey, self).__init__(basis_functions=basis_functions,
                                                   extra_features=extra_features)
        self.filtername = filtername

    def return_observations(self):
        """
        Just point at the highest reward healpix
        """
        if not self.reward_checked:
            reward = self.calc_reward_function()
        obs = empty_observation()
        # Just find the best one
        best = np.min(np.where(reward == reward.max())[0])
        ra, dec = _hpid2RaDec(best)
        obs['RA'] = ra
        obs['dec'] = dec
        obs['filtername'] = self.filtername
        obs['nexp'] = 2.
        obs['exptime'] = 30.
        return obs


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
        RA : float (0.)
            The RA of the drilling field (degrees).
        dec : float (0.)
            The Dec of the drilling field (degrees).
        """
        super(Deep_drill_survey, self).__init__(basis_functions=basis_functions,
                                                extra_features=extra_features)
        self.sequence = sequence
        self.exptime = exptime
        self.RA = np.radians(RA)
        self.dec = np.radians(dec)

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


# class Raster_survey(BaseSurvey):

