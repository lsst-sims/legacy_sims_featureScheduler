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
from lsst.sims.featureScheduler.thomson import xyz2thetaphi, thetaphi2xyz
import copy

default_nside = set_default_nside()


class BaseSurvey(object):
    def __init__(self, basis_functions, basis_weights, extra_features=None, smoothing_kernel=None,
                 ignore_obs='dummy', nside=default_nside):
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
        ignore_obs : str ('dummy')
            If an incoming observation has this string in the note, ignore it. Handy if
            one wants to ignore DD fields or observations requested by self.
        """

        if len(basis_functions) != np.size(basis_weights):
            raise ValueError('basis_functions and basis_weights must be same length.')

        # XXX-Check that input is a list of features
        self.nside = nside
        self.ignore_obs = ignore_obs
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
        if self.ignore_obs not in observation['note']:
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
            indx = np.arange(hp.nside2npix(self.nside))
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
                 smoothing_kernel=None, reward=1e6, ignore_obs='dummy',
                 nside=default_nside, min_alt=30., max_alt=85.):
        """
        min_alt : float (30.)
            The minimum altitude to attempt to chace a pair to (degrees). Default of 30 = airmass of 2.
        max_alt : float(85.)
            The maximum altitude to attempt to chase a pair to (degrees).

        """
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.nside = nside
        self.reward_val = reward
        self.reward = -reward
        if extra_features is None:
            extra_features = {'mjd': features.Current_mjd()}
            extra_features['altaz'] = features.AltAzFeature(nside=nside)
        super(Scripted_survey, self).__init__(basis_functions=basis_functions,
                                              basis_weights=basis_weights,
                                              extra_features=extra_features,
                                              smoothing_kernel=smoothing_kernel,
                                              ignore_obs=ignore_obs)

    def add_observation(self, observation, indx=None, **kwargs):
        """Check if this matches a scripted observation
        """
        # From base class
        if self.ignore_obs not in observation['note']:
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
        for key in ['RA', 'dec', 'filter', 'exptime', 'nexp', 'note']:
            observation[key] = obs_row[key]
        return observation

    def _check_alts(self, indices):
        """Check the altitudes of potential matches.
        """
        # This is kind of a kludgy low-resolution way to convert ra,dec to alt,az, but should be really fast.
        # XXX--should I stick the healpixel value on when I set the script? Might be faster.
        # XXX not sure this really needs to be it's own method
        hp_ids = _raDec2Hpid(self.nside, self.obs_wanted[indices]['RA'], self.obs_wanted[indices]['dec'])
        alts = self.extra_features['altaz'].feature['alt'][hp_ids]
        in_range = np.where((alts < self.max_alt) & (alts > self.min_alt))
        indices = indices[in_range]
        return indices

    def _check_list(self):
        """Check to see if the current mjd is good
        """
        dt = self.obs_wanted['mjd'] - self.extra_features['mjd'].feature
        # Check for matches with the right requested MJD
        matches = np.where((np.abs(dt) < self.mjd_tol) & (~self.obs_log))[0]
        # Trim down to ones that are in the altitude limits
        matches = self._check_alts(matches)
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


class Simple_greedy_survey_fields(BaseSurvey):
    """
    Chop down the reward function to just look at unmasked opsim field locations.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=25, smoothing_kernel=None, nside=default_nside, ignore_obs='ack'):
        super(Simple_greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                          basis_weights=basis_weights,
                                                          extra_features=extra_features,
                                                          smoothing_kernel=smoothing_kernel,
                                                          ignore_obs=ignore_obs)
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


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    xp = x
    yp = -y*np.cos(theta)-z*np.sin(theta)
    zp = -y*np.sin(theta)+z*np.cos(theta)
    return xp, yp, zp


class Greedy_survey_fields(BaseSurvey):
    """
    Use a field tesselation and assign each healpix to a field.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=25, smoothing_kernel=None, nside=default_nside,
                 dither=False, seed=42, ignore_obs='ack'):
        if extra_features is None:
            extra_features = {}
            extra_features['night'] = features.Current_night()
        super(Greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                   basis_weights=basis_weights,
                                                   extra_features=extra_features,
                                                   smoothing_kernel=smoothing_kernel,
                                                   ignore_obs=ignore_obs)
        self.nside = nside
        self.filtername = filtername
        # Load the OpSim field tesselation
        self.fields_init = read_fields()
        self.fields = self.fields_init.copy()
        self.block_size = block_size
        self._hp2fieldsetup(self.fields['RA'], self.fields['dec'])
        np.random.seed(seed)
        self.dither = dither
        self.night = extra_features['night'].feature + 0

    def _spin_fields(self, lon=None, lat=None):
        """Spin the field tesselation
        """
        if lon is None:
            lon = np.random.rand()*np.pi*2
        if lat is None:
            lat = np.random.rand()*np.pi*2
        # rotate longitude
        ra = (self.fields['RA'] + lon) % (2.*np.pi)
        dec = self.fields['dec'] + 0

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(self.fields['RA'], self.fields['dec']+np.pi/2.)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi/2
        ra = theta + np.pi

        self.fields['RA'] = ra
        self.fields['dec'] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more than needed.
        self._hp2fieldsetup(ra, dec)

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'update_conditions'):
                self.extra_features[feature].update_conditions(conditions, **kwargs)
        # If we are dithering and need to spin the fields
        if self.dither:
            if self.extra_features['night'].feature != self.night:
                self._spin_fields()
                self.night = self.extra_features['night'].feature + 0
        self.reward_checked = False

    def _hp2fieldsetup(self, ra, dec, leafsize=100):
        """Map each healpixel to nearest field. This will only work if healpix
        resolution is higher than field resolution.
        """
        x, y, z = treexyz(ra, dec)
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


def wrapHA(HA):
    """Make sure Hour Angle is between 0 and 24 hours """
    while HA > 24.:
        HA -= 24.
    while HA < 0:
        HA += 24.
    return HA


class Deep_drilling_survey(BaseSurvey):
    """A survey class for running deep drilling fields
    """
    def __init__(self, RA, dec, extra_features=None, sequence='gggrrriii', exptime=30.,
                 nexp=2, ignore_obs='dummy', survey_name='DD', fraction_limit=0.01,
                 HA_limits=[-1.5, 1.], reward_value=100., moon_up=True, readtime=2.):
        """
        Parameters
        ----------
        RA : float
            The RA of the field (degrees)
        dec : float
            The dec of the field to observe (degrees)
        extra_features : list of feature objects (None)
            The features to track, will construct automatically if None.
        sequence : list of observation objects or str (gggrrriii)
            The sequence of observations to take
        survey_name : str (DD)
            The name to give this survey so it can be tracked
        fraction_limit : float (0.01)
            Do not request observations if the fraction of observations from this
            survey exceeds the frac_limit.
        HA_limits : list of floats ([-1.5, 1.])
            The range of acceptable hour angles to start a sequence (hours)
        reward_value : float (100)
            The reward value to report if it is able to start (unitless)
        moon_up : bool (True)
            Require the moon to be up (True) or down (False).
        readtime : float (2.)
            Readout time for computing approximate time of observing the sequence. (seconds)
        """
        # No basis functions for this survey
        self.basis_functions = []
        self.ra = np.radians(RA)
        self.ra_hours = RA/360.*24.
        self.dec = np.radians(dec)
        self.ignore_obs = ignore_obs
        self.survey_name = survey_name
        self.HA_limits = HA_limits
        self.reward_value = reward_value
        self.moon_up = moon_up
        self.fraction_limit = fraction_limit

        if extra_features is None:
            self.extra_features = {}
            # The total number of observations
            self.extra_features['N_obs'] = features.N_obs_count()
            # The number of observations for this survey
            self.extra_features['N_obs_self'] = features.N_obs_survey(note=survey_name)
            # The current LMST. Pretty sure in hours
            self.extra_features['lmst'] = features.Current_lmst()
            # Moon altitude
            self.extra_features['sun_moon_alt'] = features.Sun_moon_alts()
            # Time to next moon rise

            # Time to twilight

            # last time this survey was observed (in case we want to force a cadence)

        else:
            self.extra_features = extra_features

        if type(sequence) == str:
            self.sequence = []
            for filtername in sequence:
                obs = empty_observation()
                obs['filter'] = filtername
                obs['exptime'] = exptime
                obs['RA'] = self.ra
                obs['dec'] = self.dec
                obs['nexp'] = nexp
                obs['note'] = survey_name
                self.sequence.append(obs)

        self.approx_time = np.sum([o['exptime']+readtime*o['nexp'] for o in obs])

    def _check_feasability(self):
        result = True
        # Check if the LMST is in range
        HA = self.extra_features['lmst'].feature - self.ra_hours
        HA = wrapHA(HA)


        if (HA < np.min(self.HA_limits)) | (HA > np.max(self.HA_limits)):
            return False
        # Check moon alt
        if self.moon_up:
            if self.extra_features['sun_moon_alt'].feature['moonAlt'] < 0.:
                return False
        else:
            if self.extra_features['sun_moon_alt'].feature['moonAlt'] > 0.:
                return False

        # Make sure twilight hasn't started
        if self.extra_features['sun_moon_alt'].feature['sunAlt'] > np.radians(-18.):
            return False

        # Check if the moon will come up
        # XXX--to do. I don't think I

        # Check if twilight starts soon
        # XXX--to do

        # Check if we are over-observed
        if self.extra_features['N_obs_self'].feature/float(self.extra_features['N_obs'].feature) > self.fraction_limit:
            return False

        # If we made it this far, good to go
        return result

    def calc_reward_function(self):
        result = -np.inf
        if self._check_feasability():
            result = self.reward_value
        return result

    def __call__(self):
        result = []
        if self._check_feasability():
            result = copy.deepcopy(self.sequence)
            # Note, could check here what the current filter is and re-order the result
        return result


class Pairs_survey_scripted(Scripted_survey):
    """Check if incoming observations will need a pair in 30 minutes. If so, add to the queue
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filt_to_pair='griz',
                 dt=40., ttol=10., reward_val=100., note='scripted', ignore_obs='ack',
                 min_alt=30., max_alt=85., lat=-30.2444, nside=default_nside):
        """
        Parameters
        ----------
        filt_to_pair : str (griz)
            Which filters to try and get pairs of
        dt : float (40.)
            The ideal gap between pairs (minutes)
        ttol : float (10.)
            The time tolerance when gathering a pair (minutes)
        """
        self.lat = np.radians(lat)
        self.note = note
        self.reward_val = reward_val
        self.ttol = ttol/60./24.
        self.dt = dt/60./24.  # To days
        if extra_features is None:
            self.extra_features = {}
            self.extra_features['Pair_map'] = features.Pair_in_night(filtername=filt_to_pair)
            self.extra_features['current_mjd'] = features.Current_mjd()
            self.extra_features['current_filter'] = features.Current_filter()
            self.extra_features['altaz'] = features.AltAzFeature(nside=nside)
            self.extra_features['current_lmst'] = features.Current_lmst()
            self.extra_features['m5_depth'] = features.M5Depth(filtername='r', nside=nside)

        super(Pairs_survey_scripted, self).__init__(basis_functions=basis_functions,
                                                    basis_weights=basis_weights,
                                                    extra_features=self.extra_features,
                                                    ignore_obs=ignore_obs, min_alt=min_alt,
                                                    max_alt=max_alt, nside=nside)
        self.filt_to_pair = filt_to_pair
        # list to hold observations
        self.observing_queue = []

    def add_observation(self, observation, indx=None, **kwargs):
        """Add an observed observation
        """

        if self.ignore_obs not in observation['note']:
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
        # Assuming self.observing_queue is sorted by MJD.
        if len(self.observing_queue) > 0:
            stale = True
            in_window = np.abs(self.observing_queue[0]['mjd']-self.extra_features['current_mjd'].feature) < self.ttol
            while stale:
                # If the next observation in queue is past the window, drop it
                if (self.observing_queue[0]['mjd'] < self.extra_features['current_mjd'].feature) & (~in_window):
                    del self.observing_queue[0]
                # If we are in the window, but masked, drop it
                elif (in_window) & (~self._check_mask(self.observing_queue[0])):
                    del self.observing_queue[0]
                else:
                    stale = False
                # If we have deleted everything, break out of where
                if len(self.observing_queue) == 0:
                    stale = False

    def _check_alts(self, observation):
        result = False
        # Just do a fast ra,dec to alt,az conversion. Can use LMST from a feature.

        alt, az = stupidFast_RaDec2AltAz(observation['RA'], observation['dec'],
                                         self.lat, None,
                                         self.extra_features['current_mjd'].feature,
                                         lmst=self.extra_features['current_lmst'].feature/12.*np.pi)
        in_range = np.where((alt < self.max_alt) & (alt > self.min_alt))[0]
        if np.size(in_range) > 0:
            result = True
        return result

    def _check_mask(self, observation):
        """Check that the proposed observation is not currently masked for some reason on the sky map.
        True if the observation is good to observe
        False if the proposed observation is masked
        """
        hpid = _raDec2Hpid(self.nside, observation['RA'], observation['dec'])[0]
        # XXX--note this is using the sky brightness. Should make features/basis functions
        # that explicitly mask moon and alt limits for clarity and use them here.
        skyval = self.extra_features['m5_depth'].feature[hpid]
        if skyval > 0:
            return True
        else:
            return False

    def calc_reward_function(self):
        self._purge_queue()
        result = -np.inf
        self.reward = result
        if len(self.observing_queue) > 0:
            # Check if the time is good and we are in a good filter.
            in_window = np.abs(self.observing_queue[0]['mjd']-self.extra_features['current_mjd'].feature) < self.ttol
            infilt = self.extra_features['current_filter'].feature in self.filt_to_pair

            if in_window & infilt:
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
            in_window = np.abs(self.observing_queue[0]['mjd']-self.extra_features['current_mjd'].feature) < self.ttol
            infilt = self.extra_features['current_filter'].feature in self.filt_to_pair
            if in_window & infilt:
                result = self.observing_queue.pop(0)
                result['note'] = self.note
                # Make sure we don't change filter if we don't have to.
                result['filter'] = self.extra_features['current_filter'].feature
                result = [result]
        return result

