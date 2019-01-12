import numpy as np
from lsst.sims.featureScheduler import features
from lsst.sims.featureScheduler import utils
import healpy as hp
import matplotlib.pylab as plt
import warnings
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ["Target_map_modulo_basis_function"]


class Target_map_modulo_basis_function(Base_basis_function):
    """Basis function that tracks number of observations and tries to match a specified spatial distribution
    can enter multiple maps that will be used at different times in the survey

    Parameters
    ----------
    day_offset : np.array
        Healpix map that has the offset to be applied to each pixel when computing what season it is on.
    filtername : (string 'r')
        The name of the filter for this target map.
    nside: int (default_nside)
        The healpix resolution.
    target_maps : list of numpy array (None)
        healpix maps showing the ratio of observations desired for all points on the sky
    norm_factor : float (0.00010519)
        for converting target map to number of observations. Should be the area of the camera
        divided by the area of a healpixel divided by the sum of all your goal maps. Default
        value assumes LSST foV has 1.75 degree radius and the standard goal maps. If using
        mulitple filters, see lsst.sims.featureScheduler.utils.calc_norm_factor for a utility
        that computes norm_factor.
    out_of_bounds_val : float (-10.)
        Reward value to give regions where there are no observations requested (unitless).
    season_modulo : int (2)
        The value to modulate the season by (years). 

    """
    def __init__(self, day_offset=None, filtername='r', nside=None, target_maps=None,
                 norm_factor=None, out_of_bounds_val=-10., season_modulo=2):

        super(Target_map_modulo_basis_function, self).__init__(nside=nside, filtername=filtername)

        if norm_factor is None:
            warnings.warn('No norm_factor set, use utils.calc_norm_factor if using multiple filters.')
            self.norm_factor = 0.00010519
        else:
            self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter

        # XXX--need to convert these features to track by season.
        self.survey_features['N_obs'] = features.N_observations(filtername=filtername, nside=self.nside)
        # Count of all the observations
        self.survey_features['N_obs_count_all'] = features.N_obs_count(filtername=None)
        if target_maps is None:
            self.target_map = utils.generate_goal_map(filtername=filtername, nside=self.nside)
        else:
            self.target_map = target_maps
        self.out_of_bounds_area = np.where(self.target_map == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

        # For computing what day each healpix is on
        if day_offset is None:
            self.day_offset = np.zeros(hp.nside2npix(self.nside), dtype=float)
        else:
            self.day_offset = day_offset

        self.season_modulo = season_modulo

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix reward map
        """

        result = self.result.copy()
        if indx is None:
            indx = self.all_indx

        # Compute what season it is at each pixel
        season = np.floor((self.day_offset[indx] + conditions.night)/365.25)
        zeroth = np.where(season < 0)[0]
        non_zero = np.where(season >= 0)[0]
        season = season % self.season_modulo

        composite_target = self.result.copy()[indx]
        composite_target[zeroth] = self.target_maps[indx][zeroth]
        for val in np.unique(season[non_zero]):
            season_indx = np.where(season == val)[0]
            composite_target[season_indx] = self.target_maps[val+1][indx][season_indx]

        # Find out how many observations we want now at those points
        goal_N = composite_target * self.survey_features['N_obs_count_all'].feature * self.norm_factor

        result[indx] = goal_N - self.survey_features['N_obs'].feature[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result
