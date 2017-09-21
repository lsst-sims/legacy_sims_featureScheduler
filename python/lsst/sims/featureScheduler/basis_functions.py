from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
from . import features
from . import utils
import healpy as hp
from lsst.sims.utils import haversine, _hpid2RaDec


default_nside = utils.set_default_nside()


class Base_basis_function(object):
    """
    Class that takes features and computes a reward fucntion when called.
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

        # Mutually exclusive sky regions
        self.SCP_indx, self.NES_indx, self.GP_indx, self.WFD_indx = utils.mutually_exclusive_regions()

    def add_observation(self, observation, indx=None):
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)

    def update_conditions(self, conditions):
        for feature in self.condition_features:
            self.condition_features[feature].update_conditions(conditions)

    def __call__(self, **kwargs):
        """
        Return a reward healpix map or a reward scalar.
        """
        pass


class Quadrant_basis_function(Base_basis_function):
    """
    """
    def __init__(self, nside=default_nside, condition_features=None, minAlt=20., maxAlt=82.,
                 azWidth=15., survey_features=None, quadrants='All'):
        if quadrants == 'All':
            self.quadrants = ['N', 'E', 'S', 'W']
        else:
            self.quadrants = quadrants
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['altaz'] = features.AltAzFeature()
        self.minAlt = np.radians(minAlt)
        self.maxAlt = np.radians(maxAlt)
        self.azWidth = np.radians(azWidth)
        self.nside = nside

    def __call__(self, indx=None):
        
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        # for now, let's just make 4 quadrants accessable. In the future
        # maybe look ahead to where the moon will be, etc

        alt = self.condition_features['altaz'].feature['alt']
        az = self.condition_features['altaz'].feature['az']

        alt_limit = np.where((alt > self.minAlt) &
                             (alt < self.maxAlt))[0]

        if 'S' in self.quadrants:
            q1 = np.where((az[alt_limit] > np.pi-self.azWidth) &
                          (az[alt_limit] < np.pi+self.azWidth))[0]
            result[alt_limit[q1]] = 1

        if 'E' in self.quadrants:
            q2 = np.where((az[alt_limit] > np.pi/2.-self.azWidth) &
                          (az[alt_limit] < np.pi/2.+self.azWidth))[0]
            result[alt_limit[q2]] = 1

        if 'W' in self.quadrants:
            q3 = np.where((az[alt_limit] > 3*np.pi/2.-self.azWidth) &
                          (az[alt_limit] < 3*np.pi/2.+self.azWidth))[0]
            result[alt_limit[q3]] = 1

        if 'N' in self.quadrants:
            q4 = np.where((az[alt_limit] < self.azWidth) |
                          (az[alt_limit] > 2*np.pi - self.azWidth))[0]
            result[alt_limit[q4]] = 1

        return result


class Target_map_basis_function(Base_basis_function):
    """
    Generate a map that rewards survey areas falling behind.
    """
    def __init__(self, filtername='r', nside=default_nside, target_map=None, softening=1.,
                 survey_features=None, condition_features=None, visits_per_point=10.,
                 out_of_bounds_val=-10.):
        """
        Parameters
        ----------
        visits_per_point : float (10.)
            How many visits can a healpixel be ahead or behind before it counts as 1 point.
        target_map : numpy array (None)
            A healpix map showing the ratio of observations desired for all points on the sky
        out_of_bounds_val : float (10.)
            Point value to give regions where there are no observations requested
        """
        if survey_features is None:
            self.survey_features = {}
            self.survey_features['N_obs'] = features.N_observations(filtername=filtername)
            self.survey_features['N_obs_reference'] = features.N_obs_reference()
        super(Target_map_basis_function, self).__init__(survey_features=self.survey_features,
                                                        condition_features=condition_features)
        self.visits_per_point = visits_per_point
        self.nside = nside
        self.softening = softening
        if target_map is None:
            self.target_map = utils.generate_goal_map(filtername=filtername)
        else:
            self.target_map = target_map
        self.out_of_bounds_area = np.where(self.target_map == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def __call__(self, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed

        Returns
        -------
        Healpix reward map
        """
        # Should probably update this to be as masked array.
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)

        # Find out how many observations we want now at those points
        scale = np.max([self.softening, self.survey_features['N_obs_reference'].feature])
        goal_N = self.target_map[indx] * scale
        result[indx] = goal_N - self.survey_features['N_obs'].feature[indx]
        result[indx] /= self.visits_per_point

        result[self.out_of_bounds_area] = self.out_of_bounds_val

        #result[indx] = -self.survey_features['N_obs'].feature[indx]
        #result[indx] /= (self.survey_features['N_obs_reference'].feature + self.softening)
        #result[indx] += self.target_map[indx]
        return result


class Obs_ratio_basis_function(Base_basis_function):
    """
    Mostly for deep_drilling fields
    """
    def __init__(self, survey_features=None, condition_features=None, ref_ra=0., ref_dec=-30.,
                 dd_ra=0., dd_dec=0., target_ratio=100., softening=1.):
        """
        blah
        """
        self.target_ratio = target_ratio
        self.softening = softening
        if survey_features is None:
            self.survey_features = {}
            self.survey_features['N_obs_reference'] = features.N_obs_reference(ra=ref_ra, dec=ref_dec)
            self.survey_features['N_obs_DD'] = features.N_obs_reference(ra=dd_ra, dec=dd_dec)

        super(Obs_ratio_basis_function, self).__init__(survey_features=self.survey_features,
                                                          condition_features=condition_features)

    def __call__(self, **kwargs):
        result = 0.
        N_DD = self.survey_features['N_obs_DD'].feature
        N_ref = self.survey_features['N_obs']
        result += self.target_ratio - (N_DD/(N_ref+self.softening))

        return result


class Spot_observable_basis_function(Base_basis_function):
    """
    Decide if a spot on the sky is good to go
    """
    def __init__(self, condition_features=None, ra=0., dec=0., lst_min=-1., lst_max=0.5):
        # Need to add sun distance requirements, moon dist requirements, seeing requirements, etc.

        # Need feature of LST.
        pass

    def __call__(self, **kwargs):
        pass


class Visit_repeat_basis_function(Base_basis_function):
    """
    Basis function to reward re-visiting an area on the sky. Looking for Solar System objects.
    """
    def __init__(self, survey_features=None, condition_features=None, gap_min=15., gap_max=45.,
                 filtername='r', nside=default_nside, npairs=1):
        """
        survey_features : dict of features (None)
            Dict of feature objects.
        gap_min : float (15.)
            Minimum time for the gap (minutes)
        gap_max : flaot (45.)
            Maximum time for a gap
        filtername : str ('r')
            The filter(s) to count with pairs
        npairs : int (1)
            The number of pairs of observations to attempt to gather
        """

        self.gap_min = gap_min/60./24.
        self.gap_max = gap_max/60./24.
        self.npairs = npairs
        self.nside = nside

        if survey_features is None:
            self.survey_features = {}
            # Track the number of pairs that have been taken in a night
            self.survey_features['Pair_in_night'] = features.Pair_in_night(filtername=filtername,
                                                                           gap_min=gap_min, gap_max=gap_max)
            # When was it last observed
            # XXX--since this feature is also in Pair_in_night, I should just access that one!
            self.survey_features['Last_observed'] = features.Last_observed(filtername=filtername)
        if condition_features is None:
            self.condition_features = {}
            # Current MJD
            self.condition_features['Current_mjd'] = features.Current_mjd()
        super(Visit_repeat_basis_function, self).__init__(survey_features=self.survey_features,
                                                          condition_features=self.condition_features)

    def __call__(self, indx=None):
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = self.condition_features['Current_mjd'].feature - self.survey_features['Last_observed'].feature[indx]
        good = np.where((diff >= self.gap_min) & (diff <= self.gap_max) &
                        (self.survey_features['Pair_in_night'].feature[indx] < self.npairs))[0]
        result[indx[good]] += 1.
        return result


class Depth_percentile_basis_function(Base_basis_function):
    """
    Return a healpix map of the reward function based on 5-sigma limiting depth percentile
    """
    def __init__(self, survey_features=None, condition_features=None, filtername='r', nside=default_nside):
        self.filtername = filtername
        self.nside = nside
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['M5Depth_percentile'] = features.M5Depth_percentile(filtername=filtername)
        super(Depth_percentile_basis_function, self).__init__(survey_features=survey_features,
                                                              condition_features=self.condition_features)

    def __call__(self, indx=None):

        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)
        if indx is None:
            indx = np.arange(result.size)
        result[indx] = self.condition_features['M5Depth_percentile'].feature[indx]
        result = ma.masked_values(result, hp.UNSEEN)
        return result


class Filter_change_basis_function(Base_basis_function):
    """
    Reward staying in the current filter.
    """
    def __init__(self, survey_features=None, condition_features=None, filtername='r'):
        self.filtername = filtername
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['Current_filter'] = features.Current_filter()
        super(Filter_change_basis_function, self).__init__(survey_features=survey_features,
                                                           condition_features=self.condition_features)

    def __call__(self, **kwargs):
        # XXX--Note here my speed observatory says None when it's parked,
        # so should be easy to start any filter. Maybe None should be reserved for no filter instead?
        if (self.condition_features['Current_filter'].feature == self.filtername) | (self.condition_features['Current_filter'].feature is None):
            result = 1.
        else:
            result = 0.
        return result


class Slewtime_basis_function(Base_basis_function):
    """Reward slews that take little time
    """
    def __init__(self, survey_features=None, condition_features=None, 
                 max_time=135., filtername='r'):
        self.maxtime = max_time
        self.filtername = filtername
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['Current_filter'] = features.Current_filter()
            self.condition_features['slewtime'] = features.SlewtimeFeature()
        super(Slewtime_basis_function, self).__init__(survey_features=survey_features,
                                                      condition_features=self.condition_features)

    def __call__(self, indx=None):
        # If we are in a different filter, the Filter_change_basis_function will take it
        if self.condition_features['Current_filter'].feature != self.filtername:
            result = 1.
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(self.condition_features['slewtime'].feature) > 1:
                result = np.zeros(np.size(self.condition_features['slewtime'].feature), dtype=float)
                good = np.where(self.condition_features['slewtime'].feature != hp.UNSEEN)
                result[good] = (self.maxtime - self.condition_features['slewtime'].feature[good])/self.maxtime
            else:
                result = (self.maxtime - self.condition_features['slewtime'].feature)/self.maxtime
        return result


class Slew_distance_basis_function(Base_basis_function):
    """
    Reward shorter slews.
    XXX-this should really be slew time, so need to break into alt and az distances.
    """
    def __init__(self, survey_features=None, condition_features=None, nside=default_nside,
                 inner_ring = 3., inner_penalty=-1., slope=-.01):
        """
        Parameters
        ----------
        inner_ring : float (3.)
            add a penalty inside this region (degrees).
        """
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['Current_pointing'] = features.Current_pointing()
        else:
            self.condition_features = condition_features
        super(Slew_distance_basis_function, self).__init__(survey_features=survey_features,
                                                           condition_features=self.condition_features)
        self.nside = nside
        self.inner_ring = np.radians(inner_ring)
        self.inner_penalty = inner_penalty
        self.slope = np.radians(slope)
        # Make the RA, Dec map
        indx = np.arange(hp.nside2npix(self.nside))
        self.ra, self.dec = _hpid2RaDec(nside, indx)

    def __call__(self, indx=None):
        if self.condition_features['Current_pointing'].feature['RA'] is None:
            return 0
        ang_distance = haversine(self.ra, self.dec, self.condition_features['Current_pointing'].feature['RA'],
                                 self.condition_features['Current_pointing'].feature['dec'])
        result = 1.+ang_distance * self.slope
        result[np.where(ang_distance <= self.inner_ring)] = self.inner_penalty
        return result



