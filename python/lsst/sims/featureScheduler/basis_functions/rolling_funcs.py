import numpy as np
from lsst.sims.featureScheduler import features
from lsst.sims.featureScheduler import utils
import healpy as hp
import matplotlib.pylab as plt
import warnings
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ["Target_map_modulo_basis_function", "Footprint_basis_function"]


class Footprint_basis_function(Base_basis_function):
    """Basis function that tries to maintain a uniformly covered footprint

    Parameters
    ----------
    filtername : str ('r')
        The filter for this footprint
    footprint : HEALpix np.array
        The desired footprint. Assumed normalized.
    all_footprints_sum : float (None)
        If using multiple filters, the sum of all the footprints. Needed to make sure basis functions are
        normalized properly across all fitlers.
    out_of_bounds_val : float (-10)
        The value to set the basis function for regions that are not in the footprint (default -10, np.nan is
        another good value to use)

    """
    def __init__(self, filtername='r', nside=None, footprint=None, all_footprints_sum=None,
                 out_of_bounds_val=-10.):

        super(Footprint_basis_function, self).__init__(nside=nside, filtername=filtername)
        self.footprint = footprint

        if all_footprints_sum is None:
            # Assume the footprints are similar in weight
            self.all_footprints_sum = np.sum(footprint)*6
        else:
            self.all_footprints_sum = all_footprints_sum

        self.footprint_sum = np.sum(footprint)

        self.survey_features = {}
        # All the observations in all filters
        self.survey_features['N_obs_all'] = features.N_observations(nside=nside, filtername=None)
        self.survey_features['N_obs'] = features.N_observations(nside=nside, filtername=filtername)

        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(footprint <= 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def _calc_value(self, conditions, indx=None):

        # Compute how many observations we should have on the sky
        desired = self.footprint / self.all_footprints_sum * np.sum(self.survey_features['N_obs_all'].feature)
        result = desired - self.survey_features['N_obs'].feature
        result[self.out_of_bounds_area] = self.out_of_bounds_val
        return result


class Footprint_rolling_basis_function(Base_basis_function):
    """Let's get the rolling really right
    """

    def __init__(self, filtername='r', nside=None, footprints=None, all_footprints_sum=None, out_of_bounds_val=-10,
                 season_modulo=2, season_length=365.25, max_season=None):
        super(Footprint_rolling_basis_function, self).__init__(nside=nside, filtername=filtername)

        # OK, going to find the parts of the map that are the same everywhere, and compute the
        # basis function the same as usual for those.
        same_footprint = np.ones(footprints[0].size, dtype=bool)
        for footprint in footprints[0:-1]:
            same_footprint = same_footprint & (footprint == footprints[-1])
        # 
        sum_footprints = footprints[0]*0
        for footprint in footprints:
            sum_footprints += footprint
        self.constant_footprint_indx = np.where((same_footprint == True) & (sum_footprints > 0))[0]
        self.rolling_footprint_indx = np.where((same_footprint == False) & (sum_footprints > 0))[0]

        self.season_modulo = season_modulo
        self.season_length = season_length
        self.max_season = max_season

        self.survey_features = {}
        # All the observations in all filters
        self.survey_features['N_obs_all'] = features.N_observations(nside=nside, filtername=None)
        # All the observations in the given filter
        self.survey_features['N_obs'] = features.N_observations(nside=nside, filtername=filtername)

        # Now I need to track the observations taken in each season.



    def _calc_value(self, conditions, indx=None):


        return result






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
        healpix maps showing the ratio of observations desired for all points on the sky. Last map will be used
        for season -1. Probably shouldn't support going to season less than -1.
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
    max_season : int (None)
        For seasons higher than this value (pre-modulo), the final target map is used.

    """
    def __init__(self, day_offset=None, filtername='r', nside=None, target_maps=None,
                 norm_factor=None, out_of_bounds_val=-10., season_modulo=2, max_season=None,
                 season_length=365.25):

        super(Target_map_modulo_basis_function, self).__init__(nside=nside, filtername=filtername)

        if norm_factor is None:
            warnings.warn('No norm_factor set, use utils.calc_norm_factor if using multiple filters.')
            self.norm_factor = 0.00010519
        else:
            self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter

        for i, temp in enumerate(target_maps[0:-1]):
            self.survey_features['N_obs_%i' % i] = features.N_observations_season(i, filtername=filtername,
                                                                                  nside=self.nside,
                                                                                  modulo=season_modulo,
                                                                                  offset=day_offset,
                                                                                  max_season=max_season,
                                                                                  season_length=season_length)
            # Count of all the observations taken in a season
            self.survey_features['N_obs_count_all_%i' % i] = features.N_obs_count_season(i, filtername=None,
                                                                                         season_modulo=season_modulo,
                                                                                         offset=day_offset,
                                                                                         nside=self.nside,
                                                                                         max_season=max_season,
                                                                                         season_length=season_length)
        # Set the final one to be -1
        self.survey_features['N_obs_%i' % -1] = features.N_observations_season(-1, filtername=filtername,
                                                                               nside=self.nside,
                                                                               modulo=season_modulo,
                                                                               offset=day_offset,
                                                                               max_season=max_season,
                                                                               season_length=season_length)
        self.survey_features['N_obs_count_all_%i' % -1] = features.N_obs_count_season(-1, filtername=None,
                                                                                      season_modulo=season_modulo,
                                                                                      offset=day_offset,
                                                                                      nside=self.nside,
                                                                                      max_season=max_season,
                                                                                      season_length=season_length)
        if target_maps is None:
            self.target_maps = utils.generate_goal_map(filtername=filtername, nside=self.nside)
        else:
            self.target_maps = target_maps
        # should probably actually loop over all the target maps?
        self.out_of_bounds_area = np.where(self.target_maps[0] == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

        # For computing what day each healpix is on
        if day_offset is None:
            self.day_offset = np.zeros(hp.nside2npix(self.nside), dtype=float)
        else:
            self.day_offset = day_offset

        self.season_modulo = season_modulo
        self.max_season = max_season
        self.season_length = season_length

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
        seasons = utils.season_calc(conditions.night, offset=self.day_offset,
                                    modulo=self.season_modulo, max_season=self.max_season,
                                    season_length=self.season_length)

        composite_target = self.result.copy()[indx]
        composite_nobs = self.result.copy()[indx]

        composite_goal_N = self.result.copy()[indx]

        for season in np.unique(seasons):
            season_indx = np.where(seasons == season)[0]
            composite_target[season_indx] = self.target_maps[season][season_indx]
            composite_nobs[season_indx] = self.survey_features['N_obs_%i' % season].feature[season_indx]
            composite_goal_N[season_indx] = composite_target[season_indx] * self.survey_features['N_obs_count_all_%i' % season].feature * self.norm_factor

        result[indx] = composite_goal_N - composite_nobs[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result
