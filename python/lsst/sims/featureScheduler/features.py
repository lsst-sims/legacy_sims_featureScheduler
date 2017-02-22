import numpy as np
import healpy as hp
from lsst.sims.utils import flat_sed_m5

# XXX-shit, should I have added _feature to all of the names here? 

# Hm, may want to pump up to nside=64.


class BaseFeature(object):
    """
    Base class for features. If a feature never cahnges, it can be a subclass of this.
    """
    def __init__(self, **kwargs):
        # self.feature should be a float, bool, or healpix size numpy array
        self.feature = None

    def __call__(self):
        return self.feature


class BaseSurveyFeature(object):
    """
    feature that tracks progreess of the survey. Takes observations and updates self.feature
    """
    def add_observation(self, observation, **kwargs):
        pass


class BaseConditionsFeature(object):
    """
    Feature based on the current conditions (e.g., mjd, cloud cover, skybrightness map, etc.)
    """
    def update_conditions(self, conditions, **kwargs):
        pass


class N_observations(BaseSurveyFeature):
    """
    Track the number of observations that have been made accross the sky
    """
    def __init__(self, filtername=None, nside=32):
        """
        Parameters
        ----------
        nside : int (32)
            The nside of the healpixel map to use
        """
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        indx : ints
            The indices of the healpixel map that have been observed by observation
        """
        if indx is None:
            # Find the hepixels that were observed by the pointing
            pass

        if (self.filtername is None) | (self.filtername == observation.filter):
            self.feature[indx] += 1


class Coadded_depth(BaseSurveyFeature):
    def __init__(self, filtername='r', nside=32):
        """
        Track the co-added depth that has been reached accross the sky
        Parameters
        ----------
        """
        self.filtername = filtername
        # Starting at limiting mag of zero should be fine.
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        """
        
        """
        if indx is None:
            # Find the hepixels that were observed by the pointing
            pass
        if observation.filter == self.filtername:
            m5 = flat_sed_m5(observation.filter, observation.skybrightness, observation.expTime,
                             observation.airmass)
            self.feature[indx] = 1.25 * np.log10(10.**(0.8*self.feature[indx]) + 10.**(0.8*m5))


class Target_frac_observations(BaseFeature):
    """
    
    """
    def __init__(self, filtername='r', nside=32, WFD_min=, WFD_max=, 
                 NES_width=20., ):
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)

        # OK, here's where we define the WFD area, NES, SCP, and Galactic plane


class Last_observed(BaseSurveyFeature):
    """
    Track when a pixel was last observed.
    """
    def __init__(self, filtername='r', nside=32):
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        if observation.filter == self.filtername:
            self.feature[indx] = observation.mjd


class N_obs_night(BaseSurveyFeature):
    """
    Track how many times something has been observed in a night
    """
    def __init__(self, filtername='r', nside=32):
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)
        self.night = -1

    def add_observation(self, observation, indx=None):
        if observation.filter == self.filtername:
            if observation.night != self.night:
                self.feature *= 0
            self.feature[indx] += 1


class N_obs_reference(BaseSurveyFeature):
    """
    Since we want to track everything by fraction, we need to declare a special spot on the sky as the
    reference point and track it independently
    """
    def __init__(self, filtername='r', ra=0., dec=-30., nside=32):
        self.filtername = filtername
        self.ra = ra
        self.dec = dec


class DD_feasability(BaseConditionsFeature):
    """
    For the DD fields, we can pre-compute hour-angles for MJD, then do a lookup to check visibility
    """



class Rotator_angle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with
    """
    def __init__(self, filtername='r', binsize=10., nside=32):
        """

        """
        self.filtername = filtername
        # Actually keep a histogram at each healpixel
        self.feature = np.zeros((hp.nside2npix(nside), 360./binsize), dtype=float)
        self.bins = np.arange(0, 360+binsize, binsize)

    def add_observation(self, observation, indx=None):
        if observation.filter == self.filtername:
            # I think this is how to broadcast things properly.
            self.feature[indx, :] += np.histogram(observation.rotSkyPos, bins=self.bins)[0]


