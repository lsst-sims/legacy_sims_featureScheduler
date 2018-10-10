from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
import healpy as hp
from lsst.sims.featureScheduler import utils
from lsst.sims.utils import m5_flat_sed



class BaseFeature(object):
    """
    Base class for features.
    """
    def __init__(self, **kwargs):
        # self.feature should be a float, bool, or healpix size numpy array, or numpy masked array
        self.feature = None

    # XXX--Should this actually be a __get__?
    def __call__(self):
        return self.feature


class BaseSurveyFeature(object):
    """
    feature that tracks progreess of the survey. Takes observations and updates self.feature
    """
    def add_observation(self, observation, **kwargs):
        raise NotImplementedError


class N_obs_count(BaseSurveyFeature):
    """Count the number of observations.
    """
    def __init__(self, filtername=None, tag=None):
        self.feature = 0
        self.filtername = filtername
        self.tag = tag

    def add_observation(self, observation, indx=None):

        if (self.filtername is None) and (self.tag is None):
            # Track all observations
            self.feature += 1
        elif (self.filtername is not None) and (self.tag is None) and (observation['filter'][0] in self.filtername):
            # Track all observations on a specified filter
            self.feature += 1
        elif (self.filtername is None) and (self.tag is not None) and (observation['tag'][0] in self.tag):
            # Track all observations on a specified tag
            self.feature += 1
        elif ((self.filtername is None) and (self.tag is not None) and
              # Track all observations on a specified filter on a specified tag
              (observation['filter'][0] in self.filtername) and (observation['tag'][0] in self.tag)):
            self.feature += 1


class N_obs_survey(BaseSurveyFeature):
    """Count the number of observations.
    """
    def __init__(self, note=None):
        """
        Parameters
        ----------
        note : str (None)
            Only count observations that have str in their note field
        """
        self.feature = 0
        self.note = note

    def add_observation(self, observation, indx=None):
        # Track all observations
        if self.note is None:
            self.feature += 1
        else:
            if self.note in observation['note']:
                self.feature += 1


class N_obs_area(BaseSurveyFeature):
    """Count the number of observations that happened on a specific region.
    """
    def __init__(self, tag_map=None):
        """
        Parameters
        ----------
        tag_map : np.ndarray (None)
            A healpix map with a tag map. Only observations with tag equals to the region tag will be counted.
        """
        self.feature = 0
        self.tag_map = tag_map

    def add_observation(self, observation, indx=None):

        if self.tag_map is not None:
            tags = np.unique(self.tag_map[indx])
            if observation['tag'] in tags:
                self.feature += 1


class Last_observation(BaseSurveyFeature):
    """When was the last observation
    """
    def __init__(self, survey_name=None):
        self.survey_name = survey_name
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observation(self, observation, indx=None):
        if self.survey_name is not None:
            if self.survey_name in observation['note']:
                self.feature = observation
        else:
            self.feature = observation


class LastSequence_observation(BaseSurveyFeature):
    """When was the last observation
    """
    def __init__(self, sequence_ids=''):
        self.sequence_ids = sequence_ids  # The ids of all sequence observations...
        # Start out with an empty observation
        self.feature = utils.empty_observation()

    def add_observation(self, observation, indx=None):
        if observation['survey_id'] in self.sequence_ids:
            self.feature = observation


class LastFilterChange(BaseSurveyFeature):
    """When was the last observation
    """
    def __init__(self):
        self.feature = {'mjd': 0.,
                        'previous_filter': None,
                        'current_filter': None}

    def add_observation(self, observation, indx=None):
        if self.feature['current_filter'] is None:
            self.feature['mjd'] = observation['mjd'][0]
            self.feature['previous_filter'] = None
            self.feature['current_filter'] = observation['filter'][0]
        elif observation['filter'][0] != self.feature['current_filter']:
            self.feature['mjd'] = observation['mjd'][0]
            self.feature['previous_filter'] = self.feature['current_filter']
            self.feature['current_filter'] = observation['filter'][0]


class N_observations(BaseSurveyFeature):
    """
    Track the number of observations that have been made accross the sky.
    """
    def __init__(self, filtername=None, nside=None, mask_indx=None):
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
        if nside is None:
            nside = utils.set_default_nside()

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
        if self.filtername is None or observation['filter'][0] in self.filtername:
            self.feature[indx] += 1

        if self.mask_indx is not None:
            overlap = np.intersect1d(indx, self.mask_indx)
            if overlap.size > 0:
                # interpolate over those pixels that are DD fields.
                # XXX.  Do I need to kdtree this? Maybe make a dict on init
                # to lookup the N closest non-masked pixels, then do weighted average.
                pass


class Coadded_depth(BaseSurveyFeature):
    def __init__(self, filtername='r', nside=None):
        """
        Track the co-added depth that has been reached accross the sky
        Parameters
        ----------
        """
        if nside is None:
            nside = utils.set_default_nside()
        self.filtername = filtername
        # Starting at limiting mag of zero should be fine.
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):

        if observation['filter'][0] == self.filtername:
            m5 = m5_flat_sed(observation['filter'][0], observation['skybrightness'][0],
                             observation['FWHMeff'][0], observation['exptime'][0],
                             observation['airmass'][0])
            self.feature[indx] = 1.25 * np.log10(10.**(0.8*self.feature[indx]) + 10.**(0.8*m5))


class Last_observed(BaseSurveyFeature):
    """
    Track when a pixel was last observed. Assumes observations are added in chronological
    order.
    """
    def __init__(self, filtername='r', nside=None):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)

    def add_observation(self, observation, indx=None):
        if self.filtername is None:
            self.feature[indx] = observation['mjd']
        elif observation['filter'][0] in self.filtername:
            self.feature[indx] = observation['mjd']


class N_obs_night(BaseSurveyFeature):
    """
    Track how many times something has been observed in a night
    (Note, even if there are two, it might not be a good pair.)
    """
    def __init__(self, filtername='r', nside=None):
        """
        Parameters
        ----------
        filtername : string ('r')
            Filter to track.
        nside : int (32)
            Scale of the healpix map
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=int)
        self.night = None

    def add_observation(self, observation, indx=None):
        if observation['night'][0] != self.night:
            self.feature *= 0
            self.night = observation['night'][0]
        if observation['filter'][0] in self.filtername:
            self.feature[indx] += 1


class Pair_in_night(BaseSurveyFeature):
    """
    Track how many pairs have been observed within a night
    """
    def __init__(self, filtername='r', nside=None, gap_min=25., gap_max=45.):
        """
        Parameters
        ----------
        gap_min : float (25.)
            The minimum time gap to consider a successful pair in minutes
        gap_max : float (45.)
            The maximum time gap to consider a successful pair (minutes)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = np.zeros(hp.nside2npix(nside), dtype=float)
        self.indx = np.arange(self.feature.size)
        self.last_observed = Last_observed(filtername=filtername)
        self.gap_min = gap_min / (24.*60)  # Days
        self.gap_max = gap_max / (24.*60)  # Days
        self.night = 0
        # Need to keep a full record of times and healpixels observed in a night.
        self.mjd_log = []
        self.hpid_log = []

    def add_observation(self, observation, indx=None):
        if observation['filter'][0] in self.filtername:
            if indx is None:
                indx = self.indx
            # Clear values if on a new night
            if self.night != observation['night']:
                self.feature *= 0.
                self.night = observation['night']
                self.mjd_log = []
                self.hpid_log = []

            # record the mjds and healpixels that were observed
            self.mjd_log.extend([np.max(observation['mjd'])]*np.size(indx))
            self.hpid_log.extend(list(indx))

            # Look for the mjds that could possibly pair with observation
            tmin = observation['mjd'] - self.gap_max
            tmax = observation['mjd'] - self.gap_min
            mjd_log = np.array(self.mjd_log)
            left = np.searchsorted(mjd_log, tmin)
            right = np.searchsorted(mjd_log, tmax, side='right')
            # Now check if any of the healpixels taken in the time gap
            # match the healpixels of the observation.
            matches = np.in1d(indx, self.hpid_log[int(left):int(right)])
            # XXX--should think if this is the correct (fastest) order to check things in.
            self.feature[indx[matches]] += 1


class Rotator_angle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with.
    XXX-under construction
    """
    def __init__(self, filtername='r', binsize=10., nside=None):
        """

        """
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        # Actually keep a histogram at each healpixel
        self.feature = np.zeros((hp.nside2npix(nside), 360./binsize), dtype=float)
        self.bins = np.arange(0, 360+binsize, binsize)

    def add_observation(self, observation, indx=None):
        if observation['filter'][0] == self.filtername:
            # I think this is how to broadcast things properly.
            self.feature[indx, :] += np.histogram(observation.rotSkyPos, bins=self.bins)[0]
