from __future__ import absolute_import
from builtins import zip
from builtins import object
import numpy as np
from .utils import empty_observation, set_default_nside, read_fields, stupidFast_altAz2RaDec, raster_sort
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site
import healpy as hp
from . import features
from . import dithering
import matplotlib.pylab as plt


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
            self.extra_features = []
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
            if hasattr(feature, 'add_observation'):
                feature.add_observation(observation, **kwargs)
        self.reward_checked = False

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            if hasattr(feature, 'update_conditions'):
                feature.update_conditions(conditions, **kwargs)
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
            self.extra_features = []
            self.extra_features.append(features.Current_mjd())
        self._set_altaz_fields()
        self.nside = nside
        self.filtername = filtername
        self.npick = npick
        site = Site(name=site)
        self.lat_rad = site.latitude_rad
        self.lon_rad = site.longitude_rad

    def _set_altaz_fields(self):
        """
        Have a fixed grid of alt,az pointings to use
        """
        tmp = read_fields()
        names = ['alt', 'az']
        types = [float, float]
        self.fields = np.zeros(tmp.size, dtype=list(zip(names, types)))
        self.fields['alt'] = tmp['dec']
        self.fields['az'] = tmp['RA']

    def _field_rewards(self):
        self.ra, self.dec = stupidFast_altAz2RaDec(self.fields['alt'], self.fields['az'],
                                                   self.lat_rad, self.lon_rad,
                                                   self.extra_features[0].feature)
        field_hpids = _raDec2Hpid(self.nside, self.ra, self.dec)
        field_rewards = self.reward[field_hpids]
        return field_rewards

    # Maybe make an alt-az tesselation, convert that to ra,dec, convert that to healpix
    # and mask everything except for those indices. Sure, why not?

    def _make_obs_list(self):
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        field_rewards = self._field_rewards()
        order = np.argsort(field_rewards)[::-1]
        # make sure we don't point at any masked pixels
        unmasked = np.where(field_rewards[order] != hp.UNSEEN)[0]
        npick = np.min([self.npick, np.max(unmasked)])
        final_ra = self.ra[order][0:npick]
        final_dec = self.dec[order][0:npick]
        final_alt = self.fields['alt'][order][0:npick]
        final_az = self.fields['az'][order][0:npick]
        # Now to sort the positions so that we raster in altitude, then az
        coords = np.empty(final_alt.size, dtype=[('alt', float), ('az', float)])
        coords['alt'] = final_alt
        coords['az'] = final_az
        indx = raster_sort(coords, order=['az', 'alt'], xbin=np.radians(5.))
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
        return observations

    def __call__(self):
        observations = self._make_obs_list()
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
                 block_size=25, smoothing_kernel=None):
        super(Simple_greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                          basis_weights=basis_weights,
                                                          extra_features=extra_features,
                                                          smoothing_kernel=smoothing_kernel)
        self.filtername = filtername
        self.fields = read_fields()
        self.field_hp = _raDec2Hpid(default_nside, self.fields['RA'], self.fields['dec'])
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


class Deep_drill_survey(BaseSurvey):
    """
    Class to make deep drilling fields.

    Rather than a single observation, the DD surveys return Scheduled Survey objects.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None,
                 RA=0, dec=0, scripted_survey=None):
        """
        Parameters
        ----------
        RA : float (0.)
            The RA of the drilling field (degrees).
        dec : float (0.)
            The Dec of the drilling field (degrees).
        scripted_survey : survey object
            A survey objcet that will return observations
        """

        # Need a basis function to see if DD is good to go

        super(Deep_drill_survey, self).__init__(basis_functions=basis_functions,
                                                basis_weights=basis_weights,
                                                extra_features=extra_features)
        self.RA = np.radians(RA)
        self.dec = np.radians(dec)

        self.scripted_survey = scripted_survey

    def __call__(self):
        
        # If there are no other scripted surveys of this type in the 
        # scripted_survey list, then send one over
        return self.scripted_survey.copy()


class Scripted_survey(BaseSurvey):
    """
    A class that will return observations from a script. And possibly self-destruct when needed.
    """
    def __init__(self):
        """
        Need to put in all the logic for if there are observations left in the sequence. 
        """

    def __call__(self):
        obs = empty_observation()
        obs['RA'] = self.RA
        obs['dec'] = self.dec
        obs['filter'] = 'z'

        return [obs]*5




############################### Cost base equivalent classes #######################################################
####################################################################################################################
class BaseSurvey_cost(object):
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
            Smooth the cost function with a Gaussian FWHM (degrees)
        """

        if len(basis_functions) != np.size(basis_weights):
            raise ValueError('basis_functions and basis_weights must be same length.')

        # XXX-Check that input is a list of features
        self.basis_functions = basis_functions
        self.basis_weights = basis_weights
        self.cost = None
        if extra_features is None:
            self.extra_features = []
        else:
            self.extra_features = extra_features
        self.cost_checked = False
        if smoothing_kernel is not None:
            self.smoothing_kernel = np.radians(smoothing_kernel)
        else:
            self.smoothing_kernel = None

        # Attribute to track if the cost function is up-to-date.
        self.cost_checked = False
        # count how many times we calc cost function
        self.cost_count = 0

    def add_observation(self, observation, **kwargs):
        for bf in self.basis_functions:
            bf.add_observation(observation, **kwargs)
        for feature in self.extra_features:
            if hasattr(feature, 'add_observation'):
                feature.add_observation(observation, **kwargs)
        self.cost_checked = False

    def update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)
        for feature in self.extra_features:
            if hasattr(feature, 'update_conditions'):
                feature.update_conditions(conditions, **kwargs)
        self.cost_checked = False

    def _check_feasability(self):
        """
        Check if the survey is feasable in the current conditions
        """
        return True

    def smooth_cost(self):
        if hp.isnpixok(self.cost.size):
            self.cost_smooth = hp.sphtfunc.smoothing(self.cost.filled(),
                                                       fwhm=self.smoothing_kernel,
                                                       verbose=False)
            good = np.where(self.cost_smooth != hp.UNSEEN)
            # Round off to prevent strange behavior early on
            self.cost_smooth[good] = np.round(self.cost_smooth[good], decimals=4)

        # Might need to check if mask expanded?

    def calc_cost_function(self):
        self.cost_count += 1
        self.cost_checked = True
        if self._check_feasability():
            indx = np.arange(hp.nside2npix(default_nside))
            self.cost = np.zeros(indx.size)
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(indx=indx)
                self.cost += basis_value*weight
                self.cost = np.where((basis_value == hp.UNSEEN), np.inf, self.cost)
                if hasattr(self.cost, 'mask'):
                    indx = np.where(self.cost.mask == False)[0]
            self.cost = np.where(self.cost == 0, np.inf, self.cost)
        else:
            # If not feasable, infinity cost
            self.cost = np.inf
        if self.smoothing_kernel is not None:
            self.smooth_cost()
            return self.cost_smooth
        else:
            return self.cost

    def __call__(self):
        """
        Returns
        -------
        one of:
            1) None
            2) A list of observations
            3) A Scripted_survey object (which can be called to return a list of observations)
        """
        # If the cost function hasn't been updated with the
        # latest info, calculate it
        if not self.cost_checked:
            self.cost = self.calc_cost_function()
        obs = empty_observation()
        return [obs]

    def viz_config(self):
        # XXX--zomg, we should have a method that goes through all the objects and
        # makes plots/prints info so there can be a little notebook showing the config!
        pass



class Simple_greedy_survey_fields_cost(BaseSurvey_cost):
    """
    Chop down the cost function to just look at unmasked opsim field locations.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=5, smoothing_kernel=None):
        super(Simple_greedy_survey_fields_cost, self).__init__(basis_functions=basis_functions,
                                                          basis_weights=basis_weights,
                                                          extra_features=extra_features,
                                                          smoothing_kernel=smoothing_kernel)
        self.filtername = filtername
        self.fields = read_fields()
        self.field_hp = _raDec2Hpid(default_nside, self.fields['RA'], self.fields['dec'])
        self.block_size = block_size

    def __call__(self):
        """
        Just point at the highest cost field
        """
        if not self.cost_checked:
            self.cost = self.calc_cost_function()
        # Let's find the best N from the fields
        cost_fields = self.cost[self.field_hp]
        #cost_fields[np.where(cost_fields.mask == True)] = -np.inf
        order = np.argsort(cost_fields)
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



class Smooth_area_survey_cost(BaseSurvey_cost):
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
            After the cost maximum is found, include any healpixels with cost value
            this percentile or higher within max_region_size
        max_area : float (160.)
            Area to try and observe per block (sq degrees).
        max_region_size : float (20.)
           How far away to consider healpixes after the cost function max is found (degrees)
        """

        # After overlap, get about 8 sq deg per pointing.

        if extra_features is None:
            self.extra_features = []
            self.extra_features.append(features.Coadded_depth(filtername=filtername,
                                                              nside=nside))
            self.extra_features[0].feature += 1e-5

        super(Smooth_area_survey_cost, self).__init__(basis_functions=basis_functions,
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
        if not self.cost_checked:
            cost_smooth = self.calc_cost_function()
        else:
            cost_smooth = self.cost_smooth

        # Pick the top healpixels to observe
        cost_min = np.where(cost_smooth == np.min(cost_smooth))[0].min()
        unmasked = np.where(self.cost_smooth != hp.UNSEEN)[0]
        selected = np.where(cost_smooth[unmasked] >= np.percentile(cost_smooth[unmasked],
                                                                     int(self.percentile_clip)))
        selected = unmasked[selected]

        to_observe = np.empty(cost_smooth.size, dtype=float)
        to_observe.fill(hp.UNSEEN)
        # Only those within max_region_size of the maximum
        max_vec = hp.pix2vec(self.nside, cost_min)
        pix_in_disk = hp.query_disc(self.nside, max_vec, self.max_region_size)

        # Select healpixels that have high cost, and are within
        # radius of the maximum pixel
        # Selected pixels are above the percentile threshold and within the radius
        selected = np.intersect1d(selected, pix_in_disk)
        if np.size(selected) > self.block_size:
            order = np.argsort(cost_smooth[selected])
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