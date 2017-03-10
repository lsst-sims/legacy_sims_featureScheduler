import numpy as np
import numpy.ma as ma
import healpy as hp
import utils
from lsst.sims.utils import m5_flat_sed, raDec2Hpid
from lsst.sims.skybrightness_pre import M5percentiles


default_nside = utils.set_default_nside()


class BaseFeature(object):
    """
    Base class for features.
    """
    def __init__(self, **kwargs):
        # self.feature should be a float, bool, or healpix size numpy array, or numpy masked array
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
    Track the number of observations that have been made accross the sky.
    """
    def __init__(self, filtername=None, nside=default_nside, mask_indx=None):
        """
        Parameters
        ----------
        filtername : str ('r')
            String or list that has all the filters that can count.
        nside : int (32)
            The nside of the healpixel map to use
        mask_indx : list of ints (None)
            List of healpixel indices to mask and interpolate over
        """
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.filtername = filtername
        self.mask_indx = mask_indx

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        indx : ints
            The indices of the healpixel map that have been observed by observation
        """

        if observation['filter'][0] in self.filtername:
            self.feature[indx] += 1

        if self.mask_indx is not None:
            overlap = np.intersect1d(indx, self.mask_indx)
            if overlap.size > 0:
                # interpolate over those pixels that are DD fields.
                # XXX.  Do I need to kdtree this? Maybe make a dict on init
                # to lookup the N closest non-masked pixels, then do weighted average.
                pass


class Coadded_depth(BaseSurveyFeature):
    def __init__(self, filtername='r', nside=default_nside):
        """
        Track the co-added depth that has been reached accross the sky
        Parameters
        ----------
        """
        self.filtername = filtername
        # Starting at limiting mag of zero should be fine.
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):

        if observation['filter'][0] == self.filtername:
            m5 = m5_flat_sed(observation['filter'], observation['skybrightness'],
                             observation['FWHMeff'], observation['expTime'],
                             observation['airmass'])
            self.feature[indx] = 1.25 * np.log10(10.**(0.8*self.feature[indx]) + 10.**(0.8*m5))


class Last_observed(BaseSurveyFeature):
    """
    Track when a pixel was last observed. Assumes observations are added in chronological
    order.
    """
    def __init__(self, filtername='r', nside=default_nside):
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        if observation['filter'][0] in self.filtername:
            self.feature[indx] = observation['mjd']


class N_obs_night(BaseSurveyFeature):
    """
    Track how many times something has been observed in a night
    (Note, even if there are two, it might not be a good pair.)
    """
    def __init__(self, filtername='r', nside=default_nside):
        """
        Parameters
        ----------
        filtername : string ('r')
            Filter to track.
        nside : int (32)
            Scale of the healpix map
        """
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)
        self.night = -1

    def add_observation(self, observation, indx=None):
        if observation.filter in self.filtername:
            if observation.night != self.night:
                self.feature *= 0
            self.feature[indx] += 1


class Pair_in_night(BaseSurveyFeature):
    """
    Track how many pairs have been observed within a night
    """
    def __init__(self, filtername='r', nside=default_nside, gap_min=15., gap_max=45.):
        """
        Parameters
        ----------
        gap_min : float (15.)
            The minimum time gap to consider a successful pair in minutes
        gap_max : float (40.)
        """
        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.indx = np.arange(self.feature.size)
        self.last_observed = Last_observed(filtername=filtername)
        self.gap_min = gap_min / (24.*60)  # Days
        self.gap_max = gap_max / (24.*60)  # Days
        self.night = 0

    def add_observation(self, observation, indx=None):
        if observation['filter'][0] in self.filtername:
            if indx is None:
                indx = self.indx
            # Clear values if on a new night
            if self.night != observation['night']:
                self.feature *= 0.
                self.night = observation['night']
            tdiff = observation['mjd'] - self.last_observed.feature[indx]
            good = np.where((tdiff >= self.gap_min) & (tdiff <= self.gap_max))[0]
            self.feature[indx[good]] += 1.
            self.last_observed.add_observation(observation, indx=indx)


class N_obs_reference(BaseSurveyFeature):
    """
    Since we want to track everything by fraction, we need to declare a special spot on the sky as the
    reference point and track it independently
    """
    def __init__(self, filtername='r', ra=0., dec=-30., nside=default_nside):
        self.feature = 0
        self.filtername = filtername
        self.ra = ra
        self.dec = dec
        # look up the healpix id of the point
        self.indx = raDec2Hpid(nside, ra, dec)

    def add_observation(self, observation, indx=None):
        if self.indx in indx:
            if observation['filter'][0] == self.filtername:
                self.feature += 1


class M5Depth_percentile(BaseConditionsFeature):
    """
    Given current conditions, return the 5-sigma limiting depth percentile map
    for a filter.
    """
    def __init__(self, filtername='r', expTime=30., nside=default_nside):
        self.filtername = filtername
        self.feature = None
        self.expTime = expTime
        self.nside = nside
        self.m5p = M5percentiles()

    def update_conditions(self, conditions):
        """
        Parameters
        ----------
        conditions : dict
            Keys should include airmass, sky_brightness, seeing.
        """
        m5 = np.empty(conditions['skybrightness'][self.filtername].size)
        m5.fill(hp.UNSEEN)
        m5_mask = np.zeros(m5.size, dtype=bool)
        m5_mask[np.where(conditions['skybrightness'][self.filtername] == hp.UNSEEN)] = True
        good = np.where(conditions['skybrightness'][self.filtername] != hp.UNSEEN)
        m5[good] = m5_flat_sed(self.filtername, conditions['skybrightness'][self.filtername][good],
                               conditions['FWHMeff'][good],
                               self.expTime, conditions['airmass'][good])

        self.feature = self.m5p.m5map2percentile(m5, filtername=self.filtername)
        self.feature[m5_mask] = hp.UNSEEN
        self.feature = hp.ud_grade(self.feature, nside_out=self.nside)
        self.feature = ma.masked_values(self.feature, hp.UNSEEN)


class Current_filter(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = conditions['filter']


class Current_mjd(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = conditions['mjd']


class Current_pointing(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = {'RA': conditions['RA'], 'dec': conditions['dec']}


class DD_feasability(BaseConditionsFeature):
    """
    For the DD fields, we can pre-compute hour-angles for MJD, then do a lookup to check visibility
    """


class Rotator_angle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with
    """
    def __init__(self, filtername='r', binsize=10., nside=default_nside):
        """

        """
        self.filtername = filtername
        # Actually keep a histogram at each healpixel
        self.feature = np.zeros((hp.nside2npix(nside), 360./binsize), dtype=float)
        self.bins = np.arange(0, 360+binsize, binsize)

    def add_observation(self, observation, indx=None):
        if observation['filter'][0] == self.filtername:
            # I think this is how to broadcast things properly.
            self.feature[indx, :] += np.histogram(observation.rotSkyPos, bins=self.bins)[0]


