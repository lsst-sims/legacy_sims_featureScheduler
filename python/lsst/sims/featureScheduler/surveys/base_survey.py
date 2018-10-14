import numpy as np
from lsst.sims.featureScheduler.utils import (empty_observation, set_default_nside,
                                              hp_in_lsst_fov, read_fields)
import healpy as hp
from lsst.sims.featureScheduler.thomson import xyz2thetaphi, thetaphi2xyz


__all__ = ['BaseSurvey', 'BaseMarkovDF_survey']


class BaseSurvey(object):

    def __init__(self, extra_features=None, extra_basis_functions=None,
                 ignore_obs='dummy', survey_name='', nside=None):
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
        extra_basis_functions : list of basis functions.
        smoothing_kernel : float (None)
            Smooth the reward function with a Gaussian FWHM (degrees)
        ignore_obs : str ('dummy')
            If an incoming observation has this string in the note, ignore it. Handy if
            one wants to ignore DD fields or observations requested by self. Take note,
            if a survey is called 'mysurvey23', setting ignore_obs to 'mysurvey2' will
            ignore it because 'mysurvey2' is a substring of 'mysurvey23'.
        """

        # XXX--nside shouldn't be in BaseSurvey. But the DDFs need it for 
        # tagging observations. TODO-take that out.
        if nside is None:
            nside = set_default_nside()

        self.nside = nside
        self.survey_name = survey_name
        self.nside = nside
        self.ignore_obs = ignore_obs

        self.reward = None
        self.sequence = False  # Specifies the survey gives sequence of observations
        self.survey_index = None

        if extra_features is None:
            self.extra_features = {}
        else:
            self.extra_features = extra_features
        if extra_basis_functions is None:
            self.extra_basis_functions = {}
        else:
            self.extra_basis_functions = extra_basis_functions
        self.reward_checked = False

        # Attribute to track if the reward function is up-to-date.
        self.reward_checked = False

    def add_observation(self, observation, **kwargs):
        # ugh, I think here I have to assume observation is an array and not a dict.
        if self.ignore_obs not in str(observation['note']):
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.extra_basis_functions:
                bf.add_observation(observation, **kwargs)
            self.reward_checked = False

    def _check_feasability(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        return True

    def calc_reward_function(self, conditions):
        """
        Parameters
        ----------
        conditions : lsst.sims.featureScheduler.features.Conditions object

        Returns
        -------
        reward : float (or array)

        """
        if self._check_feasability():
            self.reward = 0
        else:
            # If we don't pass feasability 
            self.reward = -np.inf

        self.reward_checked = True
        return self.reward

    def __call__(self, conditions):
        """
        Returns
        -------
        one of:
            1) None
            2) A list of observations
        """
        # If the reward function hasn't been updated with the
        # latest info, calculate it
        if not self.reward_checked:
            self.reward = self.calc_reward_function(conditions)
        obs = empty_observation()
        return [obs]

    def viz_config(self):
        # XXX--zomg, we should have a method that goes through all the objects and
        # makes plots/prints info so there can be a little notebook showing the config!
        pass


def rotx(theta, x, y, z):
    """rotate the x,y,z points theta radians about x axis"""
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    xp = x
    yp = y*cos_t+z*sin_t
    zp = -y*sin_t+z*cos_t
    return xp, yp, zp


class BaseMarkovDF_survey(BaseSurvey):
    """ A Markov Decision Function survey object
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None,
                 extra_basis_functions=None, smoothing_kernel=None,
                 ignore_obs='dummy', survey_name='', nside=None, seed=42,
                 dither=True, tag_fields=False, tag_map=None, tag_names=None,):

        super(BaseMarkovDF_survey, self).__init__(extra_features=extra_features,
                                                  extra_basis_functions=extra_basis_functions,
                                                  ignore_obs=ignore_obs, survey_name=survey_name,
                                                  nside=nside)

        self.basis_functions = basis_functions
        self.basis_weights = basis_weights
        # Check that weights and basis functions are same length
        if len(basis_functions) != np.size(basis_weights):
            raise ValueError('basis_functions and basis_weights must be same length.')
        # Load the OpSim field tesselation and map healpix to fields
        self.fields_init = read_fields()
        self.fields = self.fields_init.copy()
        self.hp2fields = np.array([])
        self._hp2fieldsetup(self.fields['RA'], self.fields['dec'])

        if smoothing_kernel is not None:
            self.smoothing_kernel = np.radians(smoothing_kernel)
        else:
            self.smoothing_kernel = None

        # Start tracking the night
        self.night = -1

        # Set the seed
        np.random.seed(seed)
        self.dither = dither

        # Tagging fields
        self.tag_fields = tag_fields
        if tag_fields:
            tags = np.unique(tag_map[tag_map > 0])
            for tag in tags:
                inside_tag = np.where(tag_map == tag)
                fields_id = np.unique(self.hp2fields[inside_tag])
                self.fields['tag'][fields_id] = tag
        else:
            for i in range(len(self.fields)):
                self.fields['tag'][i] = 1

    def add_observation(self, observation, **kwargs):
        """
        """
        if self.ignore_obs not in str(observation['note']):
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for feature in self.extra_features:
                self.extra_features[feature].add_observation(observation, **kwargs)
            for bf in self.extra_basis_functions:
                bf.add_observation(observation, **kwargs)
            self.reward_checked = False

    def _hp2fieldsetup(self, ra, dec, leafsize=100):
        """Map each healpixel to nearest field. This will only work if healpix
        resolution is higher than field resolution.
        """
        pointing2hpindx = hp_in_lsst_fov(nside=self.nside)
        self.hp2fields = np.zeros(hp.nside2npix(self.nside), dtype=np.int)
        for i in range(len(ra)):
            hpindx = pointing2hpindx(ra[i], dec[i])
            self.hp2fields[hpindx] = i

    def _spin_fields(self, lon=None, lat=None, lon2=None):
        """Spin the field tessellation to generate a random orientation

        The default field tesselation is rotated randomly in longitude, and then the
        pole is rotated to a random point on the sphere.

        Parameters
        ----------
        lon : float (None)
            The amount to initially rotate in longitude (radians). Will use a random value
            between 0 and 2 pi if None (default).
        lat : float (None)
            The amount to rotate in latitude (radians).
        lon2 : float (None)
            The amount to rotate the pole in longitude (radians).
        """
        if lon is None:
            lon = np.random.rand()*np.pi*2
        if lat is None:
            # Make sure latitude points spread correctly
            # http://mathworld.wolfram.com/SpherePointPicking.html
            lat = np.arccos(2.*np.random.rand() - 1.)
        if lon2 is None:
            lon2 = np.random.rand()*np.pi*2
        # rotate longitude
        ra = (self.fields_init['RA'] + lon) % (2.*np.pi)
        dec = self.fields_init['dec'] + 0

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(ra, dec+np.pi/2.)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi/2
        ra = theta + np.pi

        # One more RA rotation
        ra = (ra + lon2) % (2.*np.pi)

        self.fields['RA'] = ra
        self.fields['dec'] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more than needed.
        self._hp2fieldsetup(ra, dec)

    def smooth_reward(self):
        """If we want to smooth the reward function.
        """
        if hp.isnpixok(self.reward.size):
            self.reward_smooth = hp.sphtfunc.smoothing(self.reward.filled(),
                                                       fwhm=self.smoothing_kernel,
                                                       verbose=False)
            good = np.where(self.reward_smooth != hp.UNSEEN)
            # Round off to prevent strange behavior early on
            self.reward_smooth[good] = np.round(self.reward_smooth[good], decimals=4)

    def calc_reward_function(self, conditions):
        self.reward_checked = True
        if self._check_feasability(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            # keep track of masked pixels
            mask = np.zeros(indx.size, dtype=bool)
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                mask[np.where(basis_value == hp.UNSEEN)] = True
                if hasattr(basis_value, 'mask'):
                    mask[np.where(basis_value.mask == True)] = True
                self.reward += basis_value*weight
                # might be faster to pull this out into the feasabiliity check?
                if hasattr(self.reward, 'mask'):
                    indx = np.where(self.reward.mask == False)[0]
            self.reward[mask] = hp.UNSEEN
            # self.reward.mask = mask
            # self.reward.fill_value = hp.UNSEEN
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

    def __call__(self, conditions):

        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields()
            self.night = conditions.night.copy()

        # XXX Use self.reward to decide what to observe.
        return None