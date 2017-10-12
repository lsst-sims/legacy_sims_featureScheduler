from __future__ import absolute_import
from builtins import zip
from builtins import object
import numpy as np
from .utils import empty_observation, set_default_nside, read_fields, stupidFast_altAz2RaDec, raster_sort, stupidFast_RaDec2AltAz, gnomonic_project_toxy, treexyz
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, _angularSeparation
import healpy as hp
from . import features
from . import dithering
import matplotlib.pylab as plt
from scipy.spatial import cKDTree as kdtree
from scipy.stats import binned_statistic

default_nside = set_default_nside()


class BaseSurvey(object):
    def __init__(self, basis_functions, basis_weights, extra_features=None, smoothing_kernel=None):
        """
        Parameters
        ----------
        basis_functions : list
            List of basis_function objects
        basis_weights : numpy array
            Array the same length as basis_functions that are the
            weights to apply to each basis function
        extra_features : list XXX--should this be a dict for clarity?
            List of any additional features the survey may want to use
            e.g., for computing final dither positions, or feasability maps.
        smoothing_kernel : float (None)
            Smooth the reward function with a Gaussian FWHM (degrees)
        """

        if len(basis_functions) != np.size(basis_weights):
            raise ValueError('basis_functions and basis_weights must be same length.')

        # XXX-Check that input is a list of features
        self.basis_functions = basis_functions
        self.basis_weights = basis_weights
        self.reward = None
        if extra_features is None:
            self.extra_features = {}
        else:
            self.extra_features = extra_features
        self.reward_checked = False
        if smoothing_kernel is not None:
            self.smoothing_kernel = np.radians(smoothing_kernel)
        else:
            self.smoothing_kernel = None

        # Attribute to track if the reward function is up-to-date.
        self.reward_checked = False
        # count how many times we calc reward function
        self.reward_count = 0

    def add_observation(self, observation, **kwargs):
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'add_observation'):
                self.extra_features[feature].add_observation(observation, **kwargs)
        self.reward_checked = False

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'update_conditions'):
                self.extra_features[feature].update_conditions(conditions, **kwargs)
        self.reward_checked = False

    def _check_feasability(self):
        """
        Check if the survey is feasable in the current conditions
        """
        return True

    def smooth_reward(self):
        if hp.isnpixok(self.reward.size):
            self.reward_smooth = hp.sphtfunc.smoothing(self.reward.filled(),
                                                       fwhm=self.smoothing_kernel,
                                                       verbose=False)
            good = np.where(self.reward_smooth != hp.UNSEEN)
            # Round off to prevent strange behavior early on
            self.reward_smooth[good] = np.round(self.reward_smooth[good], decimals=4)

        # Might need to check if mask expanded?

    def calc_reward_function(self):
        self.reward_count += 1
        self.reward_checked = True
        if self._check_feasability():
            self.reward = 0
            indx = np.arange(hp.nside2npix(default_nside))
            # keep track of masked pixels
            mask = np.zeros(indx.size, dtype=bool)
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(indx=indx)
                mask[np.where(basis_value == hp.UNSEEN)] = True
                if hasattr(basis_value, 'mask'):
                    mask[np.where(basis_value.mask == True)] = True
                self.reward += basis_value*weight
                # might be faster to pull this out into the feasabiliity check?
                if hasattr(self.reward, 'mask'):
                    indx = np.where(self.reward.mask == False)[0]
            self.reward[mask] = hp.UNSEEN
            # inf reward means it trumps everything.
            if np.any(np.isinf(self.reward)):
                self.reward = np.inf

        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
        if self.smoothing_kernel is not None:
            self.smooth_reward()
            return self.reward_smooth
        else:
            return self.reward

    def __call__(self):
        """
        Returns
        -------
        one of:
            1) None
            2) A list of observations
            3) A Scripted_survey object (which can be called to return a list of observations)
        """
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        obs = empty_observation()
        return [obs]

    def viz_config(self):
        # XXX--zomg, we should have a method that goes through all the objects and
        # makes plots/prints info so there can be a little notebook showing the config!
        pass


def roundx(x, y, binstart=0.1):
    """Round off to try and grid-up nearly gridded data
    """
    bins = np.arange(x.min(), x.max()+binstart, binstart)
    counts, bin_edges = np.histogram(x, bins=bins)

    # merge together bins that are nighboring and have counts
    new_bin_edges = []
    new_bin_edges.append(bin_edges[0])
    for i, b in enumerate(bin_edges[1:]):
        if (counts[i] > 0) & (counts[i-1] > 0):
            pass
        else:
            new_bin_edges.append(bin_edges[i])
    if bin_edges[-1] != new_bin_edges[-1]:
        new_bin_edges.append(bin_edges[-1])
    indx = np.digitize(x, new_bin_edges)
    new_bin_edges = np.array(new_bin_edges)
    bin_centers = (new_bin_edges[1:]-new_bin_edges[:-1])/2. + new_bin_edges[:-1]
    new_x = bin_centers[indx-1]
    return new_x


class Scripted_survey(BaseSurvey):
    """
    Take a set of scheduled observations and serve them up.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None,
                 smoothing_kernel=None, reward=1e6):
        # All we need to know is the current time
        self.reward_val = reward
        self.reward = -reward
        if extra_features is None:
            extra_features = {'mjd': features.Current_mjd()}
        super(Scripted_survey, self).__init__(basis_functions=basis_functions,
                                              basis_weights=basis_weights,
                                              extra_features=extra_features,
                                              smoothing_kernel=smoothing_kernel)

    def add_observation(self, observation, indx=None, **kwargs):
        """Check if this matches a scripted observation
        """
        # From base class
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'add_observation'):
                self.extra_features[feature].add_observation(observation, **kwargs)
        self.reward_checked = False

        dt = self.obs_wanted['mjd'] - observation['mjd']
        # was it taken in the right time window, and hasn't already been marked as observed.
        time_matches = np.where((np.abs(dt) < self.mjd_tol) & (~self.obs_log))[0]
        for match in time_matches:
            # Might need to change this to an angular distance calc and add another tolerance?
            if (self.obs_wanted[match]['RA'] == observation['RA']) & (self.obs_wanted[match]['dec'] == observation['dec']) & (self.obs_wanted[match]['filter'] == observation['filter']):
                self.obs_log[match] = True
                break

    def calc_reward_function(self):
        """If there is an observation ready to go, execute it, otherwise, -inf
        """
        observation = self._check_list()
        if observation is None:
            self.reward = -np.inf
        else:
            self.reward = self.reward_val
        return self.reward

    def _slice2obs(self, obs_row):
        """take a slice and return a full observation object
        """
        observation = empty_observation()
        for key in ['RA', 'dec', 'filter', 'exptime', 'nexp']:
            observation[key] = obs_row[key]
        return observation

    def _check_list(self):
        """Check to see if the current mjd is good
        """
        dt = self.obs_wanted['mjd'] - self.extra_features['mjd'].feature
        matches = np.where((np.abs(dt) < self.mjd_tol) & (~self.obs_log))[0]
        if matches.size > 0:
            observation = self._slice2obs(self.obs_wanted[matches[0]])
        else:
            observation = None
        return observation

    def set_script(self, obs_wanted, mjd_tol=15.):
        """
        Parameters
        ----------
        obs_wanted : np.array
            The observations that should be executed. Needs to have columns with dtype names:
            XXX
        mjds : np.array
            The MJDs for the observaitons, should be same length as obs_list
        mjd_tol : float (15.)
            The tolerance to consider an observation as still good to observe (min)
        """
        self.mjd_tol = mjd_tol/60./24.  # to days
        self.obs_wanted = obs_wanted
        # Set something to record when things have been observed
        self.obs_log = np.zeros(obs_wanted.size, dtype=bool)

    def add_to_script(self, observation, mjd_tol=15.):
        """
        Parameters
        ----------
        observation : observation object
            The observation one would like to add to the scripted surveys
        mjd_tol : float (15.)
            The time tolerance on the observation (minutes)
        """
        self.mjd_tol = mjd_tol/60./24.  # to days
        self.obs_wanted = np.concatenate((self.obs_wanted, observation))
        self.obs_log = np.concatenate((self.obs_log, np.zeros(1, dtype=bool)))
        # XXX--could do a sort on mjd here if I thought that was a good idea.
        # XXX-note, there's currently nothing that flushes this, so adding 
        # observations can pile up nonstop. Should prob flush nightly or something

    def __call__(self):
        observation = self._check_list()
        return [observation]


class Marching_army_survey(BaseSurvey):
    """
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, smoothing_kernel=None,
                 nside=default_nside, filtername='y', npick=40, site='LSST'):
        super(Marching_army_survey, self).__init__(basis_functions=basis_functions,
                                                   basis_weights=basis_weights,
                                                   extra_features=extra_features,
                                                   smoothing_kernel=smoothing_kernel)
        if extra_features is None:
            self.extra_features = {}
            self.extra_features['mjd'] = features.Current_mjd()
        self.nside = nside
        self._set_altaz_fields()
        self.filtername = filtername
        self.npick = npick
        site = Site(name=site)
        self.lat_rad = site.latitude_rad
        self.lon_rad = site.longitude_rad

    def _set_altaz_fields(self, leafsize=10):
        """
        Have a fixed grid of alt,az pointings to use
        """
        tmp = read_fields()
        names = ['alt', 'az']
        types = [float, float]
        self.fields = np.zeros(tmp.size, dtype=list(zip(names, types)))
        self.fields['alt'] = tmp['dec']
        self.fields['az'] = tmp['RA']

        x, y, z = treexyz(self.fields['az'], self.fields['alt'])
        self.field_tree = kdtree(list(zip(x, y, z)), leafsize=leafsize,
                                 balanced_tree=False, compact_nodes=False)
        hpids = np.arange(hp.nside2npix(self.nside))
        self.reward_ra, self.reward_dec = _hpid2RaDec(self.nside, hpids)

    def _make_obs_list(self):
        """
        """
        if not self.reward_checked:
            self.reward = self.calc_reward_function()

        unmasked = np.where(self.reward != hp.UNSEEN)
        reward_alt, reward_az = stupidFast_RaDec2AltAz(self.reward_ra[unmasked],
                                                       self.reward_dec[unmasked],
                                                       self.lat_rad, self.lon_rad,
                                                       self.extra_features['mjd'].feature)
        x, y, z = treexyz(reward_az, reward_alt)

        # map the healpixels to field pointings
        dist, indx = self.field_tree.query(np.vstack((x, y, z)).T)
        field_rewards = self.reward[unmasked]

        unique_fields = np.unique(indx)
        bins = np.concatenate(([np.min(unique_fields)-1], unique_fields))+.5
        # XXX--note, this might make it possible to select a field that is in a masked region, but which
        # overlaps a healpixel that is unmasked. May need to pad out any masks my an extra half-FoV.
        field_rewards, be, bi = binned_statistic(indx, field_rewards, bins=bins, statistic='mean')

        # Ah, I can just find the distance to the max and take the nop npix
        unmasked_alt = self.fields['alt'][unique_fields]
        unmasked_az = self.fields['az'][unique_fields]

        field_max = np.max(np.where(field_rewards == np.max(field_rewards))[0])
        ang_distances = _angularSeparation(unmasked_az[field_max], unmasked_alt[field_max],
                                           unmasked_az, unmasked_alt)
        final_indx = np.argsort(ang_distances)[0:self.npick]

        final_alt = unmasked_alt[final_indx]
        final_az = unmasked_az[final_indx]

        final_ra, final_dec = stupidFast_altAz2RaDec(final_alt, final_az,
                                                     self.lat_rad, self.lon_rad,
                                                     self.extra_features['mjd'].feature)
        # Only want to send RA,Dec positions to the observatory
        # Now to sort the positions so that we raster in altitude, then az
        # if we have wrap-aroud, just project at az=0, because median will pull it the wrong way
        if final_az.max()-final_az.min() > np.pi:
            fudge = 0.
        else:
            fudge = np.median(final_az)
        coords = np.empty(final_alt.size, dtype=[('alt', float), ('az', float)])
        x, y = gnomonic_project_toxy(final_az, final_alt,
                                     fudge, np.median(final_alt))
        # Expect things to be mostly vertical in alt
        x_deg = np.degrees(x)
        x_new = roundx(x_deg, y)
        coords['alt'] = y
        coords['az'] = np.radians(x_new)

        # XXX--horrible horrible magic number xbin
        indx = raster_sort(coords, order=['az', 'alt'], xbin=np.radians(1.))
        # Now to loop over and stick all of those in a list of observations
        observations = []
        for ra, dec in zip(final_ra[indx], final_dec[indx]):
            obs = empty_observation()
            obs['RA'] = ra
            obs['dec'] = dec
            obs['filter'] = self.filtername
            obs['nexp'] = 2.
            obs['exptime'] = 30.
            observations.append(obs)
        # plt.plot(np.degrees(final_az[indx]), np.degrees(final_alt[indx]), 'o-')
        # plt.scatter(np.degrees(final_az[indx]), np.degrees(final_alt[indx]), c=field_rewards[order][0:npick][indx])

        # Could do something like look at current position and see if observation[0] or [-1] is closer to 
        # the current pointing, then reverse if needed.

        return observations

    def __call__(self):
        observations = self._make_obs_list()
        return observations


class Marching_experiment(Marching_army_survey):
    def __init__(self, basis_functions, basis_weights, extra_features=None, smoothing_kernel=None,
                 nside=default_nside, filtername='y', npick=20, site='LSST'):
        super(Marching_experiment, self).__init__(basis_functions=basis_functions,
                                                  basis_weights=basis_weights,
                                                  extra_features=extra_features,
                                                  smoothing_kernel=smoothing_kernel,
                                                  npick=npick, nside=nside, filtername=filtername)

    def __call_(self):
        observations = self._make_obs_list()
        # Only selecting 20 fields by default, so let's trace out the 20, then backtrack on them
        observations.extend(observations[::-1])
        # and now to make sure we get pairs
        observations.extend(observations)
        return observations


class Marching_army_survey_pairs(Marching_army_survey):
    """Same as marching army, only repeat the block twice
    """
    def __call__(self):
        # XXX--simple "typewriter" style where it rasters, then does
        # A long slew back to the start. Could imagine doing every-other observation, then
        # the other half in reverse order, or some other more complicated style of looping back
        observations = self._make_obs_list()
        observations.extend(observations)
        return observations


class Smooth_area_survey(BaseSurvey):
    """
    Survey that selects a large area block at a time
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 percentile_clip=90., smoothing_kernel=3.5, max_region_size=20.,
                 max_area=160., nside=default_nside):
        """
        Parameters
        ----------
        percentile_clip : 90.
            After the reward maximum is found, include any healpixels with reward value
            this percentile or higher within max_region_size
        max_area : float (160.)
            Area to try and observe per block (sq degrees).
        max_region_size : float (20.)
           How far away to consider healpixes after the reward function max is found (degrees)
        """

        # After overlap, get about 8 sq deg per pointing.

        if extra_features is None:
            self.extra_features = []
            self.extra_features.append(features.Coadded_depth(filtername=filtername,
                                                              nside=nside))
            self.extra_features[0].feature += 1e-5

        super(Smooth_area_survey, self).__init__(basis_functions=basis_functions,
                                                 basis_weights=basis_weights,
                                                 extra_features=self.extra_features,
                                                 smoothing_kernel=smoothing_kernel)
        self.filtername = filtername
        pix_area = hp.nside2pixarea(nside, degrees=True)
        block_size = int(np.round(max_area/pix_area))
        self.block_size = block_size
        # Make the dithering solving object
        self.hpc = dithering.hpmap_cross(nside=default_nside)
        self.max_region_size = np.radians(max_region_size)
        self.nside = nside
        self.percentile_clip = percentile_clip

    def __call__(self):
        """
        Return pointings for a block of sky
        """
        if not self.reward_checked:
            reward_smooth = self.calc_reward_function()
        else:
            reward_smooth = self.reward_smooth

        # Pick the top healpixels to observe
        reward_max = np.where(reward_smooth == np.max(reward_smooth))[0].min()
        unmasked = np.where(self.reward_smooth != hp.UNSEEN)[0]
        selected = np.where(reward_smooth[unmasked] >= np.percentile(reward_smooth[unmasked],
                                                                     self.percentile_clip))
        selected = unmasked[selected]

        to_observe = np.empty(reward_smooth.size, dtype=float)
        to_observe.fill(hp.UNSEEN)
        # Only those within max_region_size of the maximum
        max_vec = hp.pix2vec(self.nside, reward_max)
        pix_in_disk = hp.query_disc(self.nside, max_vec, self.max_region_size)

        # Select healpixels that have high reward, and are within
        # radius of the maximum pixel
        # Selected pixels are above the percentile threshold and within the radius
        selected = np.intersect1d(selected, pix_in_disk)
        if np.size(selected) > self.block_size:
            order = np.argsort(reward_smooth[selected])
            selected = selected[order[-self.block_size:]]

        to_observe[selected] = self.extra_features[0].feature[selected]

        # Find the pointings that observe the given pixels, and minimize the cross-correlation
        # between pointing overlaps regions and co-added depth
        self.hpc.set_map(to_observe)
        best_fit_shifts = self.hpc.minimize()
        ra_pointings, dec_pointings, obs_map = self.hpc(best_fit_shifts, return_pointings_map=True)
        # Package up the observations.
        observations = []
        for ra, dec in zip(ra_pointings, dec_pointings):
            obs = empty_observation()
            obs['RA'] = ra
            obs['dec'] = dec
            obs['filter'] = self.filtername
            obs['nexp'] = 2.
            obs['exptime'] = 30.
            observations.append(obs)
        return observations


class Simple_greedy_survey(BaseSurvey):
    """
    Just point at the healpixel with the heighest reward.
    XXX-NOTE THIS IS A BAD IDEA!
    XXX-Healpixels are NOT "evenly distributed" on the sky. Using them as pointing centers
    will result in features in the coadded depth power spectrum (I think).
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=1, smoothing_kernel=None):
        super(Simple_greedy_survey, self).__init__(basis_functions=basis_functions,
                                                   basis_weights=basis_weights,
                                                   extra_features=extra_features,
                                                   smoothing_kernel=smoothing_kernel)
        self.filtername = filtername

    def __call__(self):
        """
        Just point at the highest reward healpix
        """
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        # Just find the best one
        highest_reward = self.reward[np.where(~self.reward.mask)].max()
        best = [np.min(np.where(self.reward == highest_reward)[0])]
        # Could move this up to be a lookup rather than call every time.
        ra, dec = _hpid2RaDec(default_nside, best)
        observations = []
        for i, indx in enumerate(best):
            obs = empty_observation()
            obs['RA'] = ra[i]
            obs['dec'] = dec[i]
            obs['filter'] = self.filtername
            obs['nexp'] = 2.
            obs['exptime'] = 30.
            observations.append(obs)
        return observations


class Simple_greedy_survey_fields(BaseSurvey):
    """
    Chop down the reward function to just look at unmasked opsim field locations.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=25, smoothing_kernel=None, nside=default_nside):
        super(Simple_greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                          basis_weights=basis_weights,
                                                          extra_features=extra_features,
                                                          smoothing_kernel=smoothing_kernel)
        self.nside = nside
        self.filtername = filtername
        self.fields = read_fields()
        self.field_hp = _raDec2Hpid(self.nside, self.fields['RA'], self.fields['dec'])
        self.block_size = block_size

    def __call__(self):
        """
        Just point at the highest reward healpix
        """
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        # Let's find the best N from the fields
        reward_fields = self.reward[self.field_hp]
        reward_fields[np.where(reward_fields.mask == True)] = -np.inf
        order = np.argsort(reward_fields)[::-1]
        best_fields = order[0:self.block_size]
        observations = []
        for field in best_fields:
            obs = empty_observation()
            obs['RA'] = self.fields['RA'][field]
            obs['dec'] = self.fields['dec'][field]
            obs['filter'] = self.filtername
            obs['nexp'] = 2.
            obs['exptime'] = 30.
            observations.append(obs)
        return observations


class Greedy_survey_fields(BaseSurvey):
    """
    Chop down the reward function to just look at unmasked opsim field locations.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=25, smoothing_kernel=None, nside=default_nside):
        super(Greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                          basis_weights=basis_weights,
                                                          extra_features=extra_features,
                                                          smoothing_kernel=smoothing_kernel)
        self.nside = nside
        self.filtername = filtername
        self.fields = read_fields()
        self.field_hp = _raDec2Hpid(self.nside, self.fields['RA'], self.fields['dec'])
        self.block_size = block_size
        self._hp2fieldsetup()

    def _hp2fieldsetup(self, leafsize=100):
        """Map each healpixel to a fieldID
        """
        x, y, z = treexyz(self.fields['RA'], self.fields['dec'])
        tree = kdtree(list(zip(x, y, z)), leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        hpid = np.arange(hp.nside2npix(self.nside))
        hp_ra, hp_dec = _hpid2RaDec(self.nside, hpid)
        x, y, z = treexyz(hp_ra, hp_dec)
        d, self.hp2fields = tree.query(list(zip(x, y, z)), k=1)

    def __call__(self):
        """
        Just point at the highest reward healpix
        """
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        # Let's find the best N from the fields
        order = np.argsort(self.reward)[::-1]
        best_hp = order[0:self.block_size]
        best_fields = np.unique(self.hp2fields[best_hp])
        observations = []
        for field in best_fields:
            obs = empty_observation()
            obs['RA'] = self.fields['RA'][field]
            obs['dec'] = self.fields['dec'][field]
            obs['filter'] = self.filtername
            obs['nexp'] = 2.
            obs['exptime'] = 30.
            observations.append(obs)
        return observations


class Pairs_survey_scripted(Scripted_survey):
    """Check if incoming observations will need a pair in 30 minutes. If so, add to the queue
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filt_to_pair='griz',
                 dt=30., ttol=15., reward_val=10.):
        """
        """
        self.reward_val = reward_val
        self.ttol = ttol/60./24.
        self.dt = dt/60./24.  # To days
        if extra_features is None:
            self.extra_features = {}
            self.extra_features['Pair_map'] = features.Pair_in_night(filtername=filt_to_pair)
            self.extra_features['current_mjd'] = features.Current_mjd()

        super(Pairs_survey_scripted, self).__init__(basis_functions=basis_functions,
                                                    basis_weights=basis_weights,
                                                    extra_features=self.extra_features)
        self.filt_to_pair = filt_to_pair
        # list to hold observations
        self.observing_queue = []

    def add_observation(self, observation, indx=None, **kwargs):
        """Add an observed observation
        """

        # Update my extra features:
        for bf in self.basis_functions:
            bf.add_observation(observation, indx=indx)
        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'add_observation'):
                self.extra_features[feature].add_observation(observation, indx=indx)
        self.reward_checked = False

        # Check if this observation needs a pair
        # XXX--only supporting single pairs now. Just start up another scripted survey to grap triples, etc?
        keys_to_copy = ['RA', 'dec', 'filter', 'exptime', 'nexp']
        if (observation['filter'][0] in self.filt_to_pair) & (np.max(self.extra_features['Pair_map'].feature[indx]) < 1):
            obs_to_queue = empty_observation()
            for key in keys_to_copy:
                obs_to_queue[key] = observation[key]
            # Fill in the ideal time we would like this observed
            obs_to_queue['mjd'] = observation['mjd'] + self.dt
            self.observing_queue.append(obs_to_queue)

    def _purge_queue(self):
        """Remove any pair where it's too late to observe it
        """
        if len(self.observing_queue) > 0:
            stale = True
            while stale:
                if self.observing_queue[0]['mjd'] > (self.extra_features['current_mjd'].feature + self.dt + self.ttol):
                    del self.observing_queue[0]
                else:
                    stale = False
                if len(self.observing_queue) == 0:
                    stale = False

    def calc_reward_function(self):
        self._purge_queue()
        result = -np.inf
        self.reward = result
        if len(self.observing_queue) > 0:
            if (self.observing_queue[0]['mjd'] > (self.extra_features['current_mjd'].feature - self.ttol)) & (self.observing_queue[0]['mjd'] < (self.extra_features['current_mjd'].feature + self.ttol)):
                result = self.reward_val
                self.reward = self.reward_val
        self.reward_checked = True
        return result

    def __call__(self):
        # Toss anything in the queue that is too old to pair up:
        self._purge_queue()
        # Check for something I want a pair of
        result = []
        if len(self.observing_queue) > 0:
            if (self.observing_queue[0]['mjd'] > (self.extra_features['current_mjd'].feature - self.ttol)) & (self.observing_queue[0]['mjd'] < (self.extra_features['current_mjd'].feature + self.ttol)):
                result = self.observing_queue.pop(0)
                result['note'] = 'scripted'
                result = [result]
        return result

