from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
from lsst.sims.featureScheduler import features
from lsst.sims.featureScheduler import utils
import healpy as hp
from lsst.sims.skybrightness_pre import M5percentiles
import matplotlib.pylab as plt


class Base_basis_function(object):
    """
    Class that takes features and computes a reward function when called.
    """

    def __init__(self, nside=None, filtername=None, **kwargs):
        """
        """

        # Set if basis function needs to be recalculated if there is a new observation
        self.update_on_newobs = True
        # Set if basis function needs to be recalculated if conditions change
        self.update_on_mjd = True
        # Dict to hold all the features we want to track
        self.survey_features = {}
        # Keep track of the last time the basis function was called. If mjd doesn't change, use cached value
        self.mjd_last = None
        self.value = None
        # list the attributes to compare to check if basis functions are equal.
        self.attrs_to_compare = []
        # Do we need to recalculate the basis function
        self.recalc = True
        # Basis functions don't technically all need an nside, but so many do might as well set it here
        if nside is None:
            self.nside = utils.set_default_nside()
        else:
            self.nside = nside

        self.filtername = filtername

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : np.array
            An array with information about the input observation
        indx : np.array
            The indices of the healpix map that the observation overlaps with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)
        if self.update_on_newobs:
            self.recalc = True

    def check_feasibility(self, conditions):
        """XXX--might not need this here since surveys can check feasibility?
        """
        return True

    def _calc_value(self, conditions, **kwarge):
        self.value = None
        # Update the last time we had an mjd
        self.mjd_last = conditions.mjd + 0
        self.recalc = False

    def __eq__(self):
        # XXX--to work on if we need to make a registry of basis functions.
        pass

    def __ne__(self):
        pass

    def __call__(self, conditions, **kwargs):
        """
        Parameters
        ----------
        conditions : lsst.sims.featureScheduler.features.conditions object
             Object that has attributes for all the current conditions.

        Return a reward healpix map or a reward scalar.
        """
        if self.recalc:
            self.value = self._calc_value(conditions, **kwargs)
        if self.update_on_mjd:
            if conditions.mjd != self.mjd_last:
                self.value = self._calc_value(conditions, **kwargs)
        return self.value


class Constant_basis_function(Base_basis_function):
    """Just add a constant
    """
    def __call__(self, **kwargs):
        return 1


class Target_map_basis_function(Base_basis_function):
    """Basis function that tracks number of observations and tries to match a specified spatial distribution
    """
    def __init__(self, filtername='r', nside=None, target_map=None,
                 norm_factor=0.00010519,
                 out_of_bounds_val=-10.):
        """
        Parameters
        ----------
        filtername: (string 'r')
            The name of the filter for this target map.
        nside: int (default_nside)
            The healpix resolution.
        target_map : numpy array (None)
            A healpix map showing the ratio of observations desired for all points on the sky
        norm_factor : float (0.00010519)
            for converting target map to number of observations. Should be the area of the camera
            divided by the area of a healpixel divided by the sum of all your goal maps. Default
            value assumes LSST foV has 1.75 degree radius and the standard goal maps.
        out_of_bounds_val : float (-10.)
            Point value to give regions where there are no observations requested
        """

        super(Target_map_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.norm_factor = norm_factor

        self.survey_features = {}
        # Map of the number of observations in filter
        self.survey_features['N_obs'] = features.N_observations(filtername=filtername, nside=self.nside)
        # Count of all the observations
        self.survey_features['N_obs_count_all'] = features.N_obs_count(filtername=None)
        if target_map is None:
            self.target_map = utils.generate_goal_map(filtername=filtername, nside=self.nside)
        else:
            self.target_map = target_map
        self.out_of_bounds_area = np.where(self.target_map == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.all_indx = np.arange(self.result.size)

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
        # Should probably update this to be as masked array.
        result = self.result.copy()
        if indx is None:
            indx = self.all_indx

        # Find out how many observations we want now at those points
        goal_N = self.target_map[indx] * self.survey_features['N_obs_count_all'].feature * self.norm_factor

        result[indx] = goal_N - self.survey_features['N_obs'].feature[indx]
        result[self.out_of_bounds_area] = self.out_of_bounds_val

        return result


class Avoid_Fast_Revists(Base_basis_function):
    """Marks targets as unseen if they are in a specified time window in order to avoid fast revisits.
    """
    def __init__(self, filtername='r', nside=None, gap_min=25.,
                 penalty_val=hp.UNSEEN):
        """
        Parameters
        ----------
        filtername: (string 'r')
            The name of the filter for this target map.
        gap_min : float (25.)
            Minimum time for the gap (minutes).
        nside: int (default_nside)
            The healpix resolution.
        """
        super(Avoid_Fast_Revists, self).__init__(nside=nside, filtername=filtername)

        self.filtername = filtername
        self.penalty_val = penalty_val

        self.gap_min = gap_min/60./24.
        self.nside = nside

        self.survey_features = dict()
        self.survey_features['Last_observed'] = features.Last_observed(filtername=filtername, nside=nside)

    def _calc_value(self, conditions, indx=None):
        result = np.ones(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = conditions.mjd - self.survey_features['Last_observed'].feature[indx]
        bad = np.where(diff < self.gap_min)[0]
        result[indx[bad]] = self.penalty_val
        return result


class Visit_repeat_basis_function(Base_basis_function):
    """
    Basis function to reward re-visiting an area on the sky. Looking for Solar System objects.
    """
    def __init__(self, gap_min=25., gap_max=45.,
                 filtername='r', nside=None, npairs=1):
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
        super(Visit_repeat_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.gap_min = gap_min/60./24.
        self.gap_max = gap_max/60./24.
        self.npairs = npairs

        self.survey_features = {}
        # Track the number of pairs that have been taken in a night
        self.survey_features['Pair_in_night'] = features.Pair_in_night(filtername=filtername,
                                                                       gap_min=gap_min, gap_max=gap_max,
                                                                       nside=nside)
        # When was it last observed
        # XXX--since this feature is also in Pair_in_night, I should just access that one!
        self.survey_features['Last_observed'] = features.Last_observed(filtername=filtername,
                                                                       nside=nside)

    def _calc_value(self, conditions, indx=None):
        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = conditions.mjd - self.survey_features['Last_observed'].feature[indx]
        good = np.where((diff >= self.gap_min) & (diff <= self.gap_max) &
                        (self.survey_features['Pair_in_night'].feature[indx] < self.npairs))[0]
        result[indx[good]] += 1.
        return result


class M5_diff_basis_function(Base_basis_function):
    """Basis function based on the 5-sigma depth.
    Look up the best a pixel gets, and compute the limiting depth difference with current conditions
    """
    def __init__(self, filtername='r', nside=None):
        """
        """
        super(M5_diff_basis_function, self).__init__(nside=nside, filtername=filtername)
        # Need to look up the deepest m5 values for all the healpixels
        m5p = M5percentiles()
        self.dark_map = m5p.dark_map(filtername=filtername, nside_out=self.nside)

    def add_observation(self, observation, indx=None):
        # No tracking of observations in this basis function. Purely based on conditions.
        pass

    def _calc_value(self, conditions, indx=None):
        # No way to get the sign on this right the first time.
        result = conditions.M5Depth[self.filtername] - self.dark_map
        mask = np.where(conditions.M5Depth[self.filtername].filled() == hp.UNSEEN)
        result[mask] = hp.UNSEEN
        return result


class Strict_filter_basis_function(Base_basis_function):
    """Remove the bonus for staying in the same filter if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense to consider
    a filter change. This basis function rewards if it matches the current filter, the moon rises or sets,
    twilight starts or stops, or there has been a large gap since the last observation.

    """
    def __init__(self, time_lag=10., filtername='r', twi_change=-18.):
        """
        Paramters
        ---------
        time_lag : float (10.)
            If there is a gap between observations longer than this, let the filter change (minutes)
        twi_change : float (-18.)
            The sun altitude to consider twilight starting/ending
        """
        super(Strict_filter_basis_function, self).__init__(filtername=filtername)

        self.time_lag = time_lag/60./24.  # Convert to days
        self.twi_change = np.radians(twi_change)

        self.survey_features = {}
        self.survey_features['Last_observation'] = features.Last_observation()

    def _calc_value(self, conditions, **kwargs):
        # Did the moon set or rise since last observation?
        moon_changed = conditions.moonAlt * self.survey_features['Last_observation'].feature['moonAlt'] < 0

        # Are we already in the filter (or at start of night)?
        in_filter = (conditions.current_filter == self.filtername) | (conditions.current_filter is None)

        # Has enough time past?
        time_past = (conditions.mjd - self.survey_features['Last_observation'].feature['mjd']) > self.time_lag

        # Did twilight start/end?
        twi_changed = (conditions.sunAlt - self.twi_change) * (self.survey_features['Last_observation'].feature['sunAlt']- self.twi_change) < 0

        # Did we just finish a DD sequence
        wasDD = self.survey_features['Last_observation'].feature['note'] == 'DD'

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | in_filter | time_past | twi_changed | wasDD) & mounted:
            result = 1.
        else:
            result = 0.

        return result


class Goal_Strict_filter_basis_function(Base_basis_function):
    """Remove the bonus for staying in the same filter if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense to consider
    a filter change. This basis function rewards if it matches the current filter, the moon rises or sets,
    twilight starts or stops, or there has been a large gap since the last observation.

    """

    def __init__(self, time_lag_min=10., time_lag_max=30.,
                 time_lag_boost=60., boost_gain=2.0, unseen_before_lag=False,
                 filtername='r', tag=None, twi_change=-18., proportion=1.0, aways_available=False):
        """
        Parameters
        ---------
        time_lag_min: Minimum time after a filter change for which a new filter change will receive zero reward, or
            be denied at all (see unseen_before_lag).
        time_lag_max: Time after a filter change where the reward for changing filters achieve its maximum.
        time_lag_boost: Time after a filter change to apply a boost on the reward.
        boost_gain: A multiplier factor for the reward after time_lag_boost.
        unseen_before_lag: If True will make it impossible to switch filter before time_lag has passed.
        filtername: The filter for which this basis function will be used.
        tag: When using filter proportion use only regions with this tag to count for observations.
        twi_change: Switch reward on when twilight changes.
        proportion: The expected filter proportion distribution.
        aways_available: If this is true the basis function will aways be computed regardless of the feasibility. If
            False a more detailed feasibility check is performed. When set to False, it may speed up the computation
            process by avoiding to compute the reward functions paired with this bf, when observation is not feasible.
        """
        super(Goal_Strict_filter_basis_function, self).__init__(filtername=filtername)

        self.time_lag_min = time_lag_min / 60. / 24.  # Convert to days
        self.time_lag_max = time_lag_max / 60. / 24.  # Convert to days
        self.time_lag_boost = time_lag_boost / 60. / 24.
        self.boost_gain = boost_gain
        self.unseen_before_lag = unseen_before_lag

        self.twi_change = np.radians(twi_change)
        self.proportion = proportion
        self.aways_available = aways_available

        self.survey_features = {}
        self.survey_features['Last_observation'] = features.Last_observation()
        self.survey_features['Last_filter_change'] = features.LastFilterChange()
        self.survey_features['N_obs_all'] = features.N_obs_count(filtername=None)
        self.survey_features['N_obs'] = features.N_obs_count(filtername=filtername,
                                                             tag=tag)

    def filter_change_bonus(self, time):

        lag_min = self.time_lag_min
        lag_max = self.time_lag_max

        a = 1. / (lag_max - lag_min)
        b = -a * lag_min

        bonus = a * time + b
        # How far behind we are with respect to proportion?
        nobs = self.survey_features['N_obs'].feature
        nobs_all = self.survey_features['N_obs_all'].feature
        goal = self.proportion
        # need = 1. - nobs / nobs_all + goal if nobs_all > 0 else 1. + goal
        need = goal / nobs * nobs_all if nobs > 0 else 1.
        # need /= goal
        if hasattr(time, '__iter__'):
            before_lag = np.where(time <= lag_min)
            bonus[before_lag] = -np.inf if self.unseen_before_lag else 0.
            after_lag = np.where(time >= lag_max)
            bonus[after_lag] = 1. if time < self.time_lag_boost else self.boost_gain
        elif time <= lag_min:
            return -np.inf if self.unseen_before_lag else 0.
        elif time >= lag_max:
            return 1. if time < self.time_lag_boost else self.boost_gain

        return bonus * need

    def check_feasibility(self, conditions):
        """
        This method makes a pre-check of the feasibility of this basis function. If a basis function return False
        on the feasibility check, it won't computed at all.

        :return:
        """

        # Make a quick check about the feasibility of this basis function. If current filter is none, telescope
        # is parked and we could, in principle, switch to any filter. If this basis function computes reward for
        # the current filter, then it is also feasible. At last we check for an "aways_available" flag. Meaning, we
        # force this basis function to be aways be computed.
        if conditions.current_filter is None or conditions.current_filter == self.filtername or self.aways_available:
            return True

        # If we arrive here, we make some extra checks to make sure this bf is feasible and should be computed.

        # Did the moon set or rise since last observation?
        moon_changed = conditions.moonAlt * self.survey_features['Last_observation'].feature['moonAlt'] < 0

        # Are we already in the filter (or at start of night)?
        not_in_filter = (conditions.current_filter != self.filtername)

        # Has enough time past?
        lag = conditions.mjd - self.survey_features['Last_filter_change'].feature['mjd']
        time_past = lag > self.time_lag_min

        # Did twilight start/end?
        twi_changed = (conditions.sunAlt - self.twi_change) * \
                      (self.survey_features['Last_observation'].feature['sunAlt'] - self.twi_change) < 0

        # Did we just finish a DD sequence
        wasDD = self.survey_features['Last_observation'].feature['note'] == 'DD'

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | time_past | twi_changed | wasDD) & mounted & not_in_filter:
            return True
        else:
            return False

    def _calc_value(self, conditions, **kwargs):

        if conditions.current_filter is None:
            return 0.  # no bonus if no filter is mounted
        # elif self.condition_features['Current_filter'].feature == self.filtername:
        #     return 0.  # no bonus if on the filter already

        # Did the moon set or rise since last observation?
        moon_changed = conditions.moonAlt * \
                       self.survey_features['Last_observation'].feature['moonAlt'] < 0

        # Are we already in the filter (or at start of night)?
        # not_in_filter = (self.condition_features['Current_filter'].feature != self.filtername)

        # Has enough time past?
        lag = conditions.mjd - self.survey_features['Last_filter_change'].feature['mjd']
        time_past = lag > self.time_lag_min

        # Did twilight start/end?
        twi_changed = (conditions.sunAlt - self.twi_change) * (
                    self.survey_features['Last_observation'].feature['sunAlt'] - self.twi_change) < 0

        # Did we just finish a DD sequence
        wasDD = self.survey_features['Last_observation'].feature['note'] == 'DD'

        # Is the filter mounted?
        mounted = self.filtername in conditions.mounted_filters

        if (moon_changed | time_past | twi_changed | wasDD) & mounted:
            result = self.filter_change_bonus(lag) if time_past else 0.
        else:
            result = -100. if self.unseen_before_lag else 0.

        return result


class Filter_change_basis_function(Base_basis_function):
    """
    Reward staying in the current filter.
    """
    def __init__(self, filtername='r'):
        super(Filter_change_basis_function, self).__init__(filtername=filtername)

    def _calc_value(self, conditions, **kwargs):

        if (conditions.current_filter == self.filtername) | (conditions.current_filter is None):
            result = 1.
        else:
            result = 0.
        return result


class Slewtime_basis_function(Base_basis_function):
    """Reward slews that take little time
    """
    def __init__(self, max_time=135., filtername='r', nside=None):
        super(Slewtime_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.maxtime = max_time
        self.nside = nside
        self.filtername = filtername
        self.result = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        # No tracking of observations in this basis function. Purely based on conditions.
        pass

    def _calc_value(self, conditions, indx=None):
        # If we are in a different filter, the Filter_change_basis_function will take it
        if conditions.current_filter != self.filtername:
            result = 0.
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(conditions.slewtime) > 1:
                result = self.result.copy()
                good = np.where(conditions.slewtime != hp.UNSEEN)
                result[good] = (self.maxtime - conditions.slewtime[good])/self.maxtime
            else:
                result = (self.maxtime - conditions.slewtime)/self.maxtime
        return result


class Aggressive_Slewtime_basis_function(Base_basis_function):
    """Reward slews that take little time

    XXX--not sure how this is different from Slewtime_basis_function. 
    Looks like it's checking the slewtime to the field position rather than the healpix maybe?
    """

    def __init__(self, max_time=135., order=1., hard_max=None, filtername='r', nside=None):
        super(Aggressive_Slewtime_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.maxtime = max_time
        self.hard_max = hard_max
        self.order = order
        self.result = np.zeros(hp.nside2npix(nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        # If we are in a different filter, the Filter_change_basis_function will take it
        if conditions.current_filter != self.filtername:
            result = 0.
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(self.condition_features['slewtime'].feature) > 1:
                result = self.result.copy()
                result.fill(hp.UNSEEN)

                good = np.where(np.bitwise_and(conditions.slewtime > 0.,
                                               conditions.slewtime < self.maxtime))
                result[good] = ((self.maxtime - conditions.slewtime[good]) /
                                self.maxtime) ** self.order
                if self.hard_max is not None:
                    not_so_good = np.where(conditions.slewtime > self.hard_max)
                    result[not_so_good] -= 10.
                fields = np.unique(conditions.hp2fields[good])
                for field in fields:
                    hp_indx = np.where(conditions.hp2fields == field)
                    result[hp_indx] = np.min(result[hp_indx])
            else:
                result = (self.maxtime - conditions.slewtime) / self.maxtime
        return result


class Skybrightness_limit_basis_function(Base_basis_function):
    """mask regions that are outside a sky brightness limit

    """
    def __init__(self, nside=None, filtername='r', min=20., max=30.):
        """
        Parameters
        moon_distance: float (30.)
            Minimum allowed moon distance. (degrees)
        """
        super(Skybrightness_limit_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.min = min
        self.max = max
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(hp.UNSEEN)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        good = np.where(np.bitwise_and(conditions.skybrightness[self.filtername] > self.min,
                                       conditions.skybrightness[self.filtername] < self.max))
        result[good] = 1.0

        return result


class CableWrap_unwrap_basis_function(Base_basis_function):
    """
    """
    def __init__(self, nside=None, minAz=-270., maxAz=270., minAlt=20., maxAlt=82.,
                 activate_tol=20., delta_unwrap=1.2, unwrap_until=70., max_duration=30.):
        """
        Parameters
        ----------
        minAz : float (20.)
            The minimum azimuth to activate bf (degrees)
        maxAz : float (82.)
            The maximum azimuth to activate bf (degrees)
        unwrap_until: float (90.)
            The window in which the bf is activated (degrees)
        """
        super(CableWrap_unwrap_basis_function, self).__init__(nside=nside)

        self.minAz = np.radians(minAz)
        self.maxAz = np.radians(maxAz)

        self.activate_tol = np.radians(activate_tol)
        self.delta_unwrap = np.radians(delta_unwrap)
        self.unwrap_until = np.radians(unwrap_until)

        self.minAlt = np.radians(minAlt)
        self.maxAlt = np.radians(maxAlt)
        # Convert to half-width for convienence
        self.nside = nside
        self.active = False
        self.unwrap_direction = 0.  # either -1., 0., 1.
        self.max_duration = max_duration/60./24.  # Convert to days
        self.activation_time = None
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()

        current_abs_rad = np.radians(conditions.az)
        unseen = np.where(np.bitwise_or(conditions.alt < self.minAlt,
                                        conditions.alt > self.maxAlt))
        result[unseen] = hp.UNSEEN

        if (self.minAz + self.activate_tol < current_abs_rad < self.maxAz - self.activate_tol) and not self.active:
            return result
        elif self.active and self.unwrap_direction == 1 and current_abs_rad > self.minAz+self.unwrap_until:
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result
        elif self.active and self.unwrap_direction == -1 and current_abs_rad < self.maxAz-self.unwrap_until:
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result
        elif (self.activation_time is not None and
              conditions.mjd - self.activation_time > self.max_duration):
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result

        if not self.active:
            self.activation_time = conditions.mjd
            if current_abs_rad < 0.:
                self.unwrap_direction = 1  # clock-wise unwrap
            else:
                self.unwrap_direction = -1  # counter-clock-wise unwrap

        self.active = True

        max_abs_rad = self.maxAz
        min_abs_rad = self.minAz

        TWOPI = 2.*np.pi

        # Compute distance and accumulated az.
        norm_az_rad = np.divmod(conditions.az - min_abs_rad, TWOPI)[1] + min_abs_rad
        distance_rad = divmod(norm_az_rad - current_abs_rad, TWOPI)[1]
        get_shorter = np.where(distance_rad > np.pi)
        distance_rad[get_shorter] -= TWOPI
        accum_abs_rad = current_abs_rad + distance_rad

        # Compute wrap regions and fix distances
        mask_max = np.where(accum_abs_rad > max_abs_rad)
        distance_rad[mask_max] -= TWOPI
        mask_min = np.where(accum_abs_rad < min_abs_rad)
        distance_rad[mask_min] += TWOPI

        # Step-2: Repeat but now with compute reward to unwrap using specified delta_unwrap
        unwrap_current_abs_rad = current_abs_rad - (np.abs(self.delta_unwrap) if self.unwrap_direction > 0
            else -np.abs(self.delta_unwrap))
        unwrap_distance_rad = divmod(norm_az_rad - unwrap_current_abs_rad, TWOPI)[1]
        unwrap_get_shorter = np.where(unwrap_distance_rad > np.pi)
        unwrap_distance_rad[unwrap_get_shorter] -= TWOPI
        unwrap_distance_rad = np.abs(unwrap_distance_rad)

        if self.unwrap_direction < 0:
            mask = np.where(accum_abs_rad > unwrap_current_abs_rad)
        else:
            mask = np.where(accum_abs_rad < unwrap_current_abs_rad)

        # Finally build reward map
        result = (1. - unwrap_distance_rad/np.max(unwrap_distance_rad))**2.
        result[mask] = 0.
        result[unseen] = hp.UNSEEN

        return result


class Cadence_enhance_basis_function(Base_basis_function):
    """Drive a certain cadence"""
    def __init__(self, filtername='gri', nside=None,
                 supress_window=[0, 1.8], supress_val=-0.5,
                 enhance_window=[2.1, 3.2], enhance_val=1.,
                 apply_area=None):
        """
        Parameters
        ----------
        filtername : str ('gri')
            The filter(s) that should be grouped together
        supress_window : list of float
            The start and stop window for when observations should be repressed (days)
        apply_area : healpix map
            The area over which to try and drive the cadence. Good values as 1, no candece drive 0.
            Probably works as a bool array too.
        """
        super(Cadence_enhance_basis_function, self).__init__(nside=nside, filtername=filtername)

        self.supress_window = np.sort(supress_window)
        self.supress_val = supress_val
        self.enhance_window = np.sort(enhance_window)
        self.enhance_val = enhance_val

        survey_features = {}
        survey_features['last_observed'] = features.Last_observed(filtername=filtername)

        self.empty = np.zeros(hp.nside2npix(self.nside), dtype=float)
        # No map, try to drive the whole area
        if apply_area is None:
            self.apply_indx = np.arange(self.empty.size)
        else:
            self.apply_indx = np.where(apply_area != 0)[0]

    def _calc_value(self, conditions, indx=None):
        # copy an empty array
        result = self.empty.copy()
        if indx is not None:
            ind = np.intersect1d(indx, self.apply_indx)
        else:
            ind = self.apply_indx
        if np.size(ind) == 0:
            result = 0
        else:
            mjd_diff = conditions.mjd - self.survey_features['last_observed'].feature[ind]
            to_supress = np.where((mjd_diff > self.supress_window[0]) & (mjd_diff < self.supress_window[1]))
            result[ind[to_supress]] = self.supress_val
            to_enhance = np.where((mjd_diff > self.enhance_window[0]) & (mjd_diff < self.enhance_window[1]))
            result[ind[to_enhance]] = self.enhance_val
        return result
