from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
import healpy as hp
from . import utils
from lsst.sims.utils import m5_flat_sed, raDec2Hpid, Site, _hpid2RaDec


default_nside = None


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


class BaseConditionsFeature(object):
    """
    Feature based on the current conditions (e.g., mjd, cloud cover, skybrightness map, etc.)
    """
    def update_conditions(self, conditions, **kwargs):
        raise NotImplementedError


class AltAzFeature(BaseConditionsFeature):
    """Compute the alt and az of all the pixels
    """
    def __init__(self, nside=default_nside, lat=None, lon=None):
        """
        Parameters
        ----------
        nside : int (32)
            The nside of the healpixel map to use
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.ra, self.dec = utils.ra_dec_hp_map(nside)
        if lat is None:
            site = Site(name='LSST')
            self.lat = site.latitude_rad
            self.lon = site.longitude_rad
        else:
            self.lat = lat
            self.lon = lon

    def update_conditions(self, conditions, *kwargs):
        self.feature = {}
        alt, az = utils.stupidFast_RaDec2AltAz(self.ra, self.dec,
                                               self.lat, self.lon, conditions['mjd'])
        self.feature['alt'] = alt
        self.feature['az'] = az


class BulkCloudCover(BaseConditionsFeature):

    def __init__(self):
        """Store the bulk cloud cover.
        """
        self.feature = dict()

    def update_conditions(self, conditions, *kwargs):
        self.feature = conditions['clouds']


class N_obs_count(BaseSurveyFeature):
    """Count the number of observations.
    """
    def __init__(self, filtername=None):
        self.feature = 0
        self.filtername = filtername

    def add_observation(self, observation, indx=None):
        # Track all observations
        if self.filtername is None:
            self.feature += 1
        else:
            if observation['filter'][0] in self.filtername:
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
    def __init__(self, filtername='r', nside=default_nside):
        if nside is None:
            nside = utils.set_default_nside()

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
    def __init__(self, filtername='r', nside=default_nside, gap_min=25., gap_max=45.):
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


class SlewtimeFeature(BaseConditionsFeature):
    """Grab the slewtime map from the observatory.

    The slewtimes are expected to be a healpix map, in a single filter.
    The time to change filters is included in a feature
    """
    def __init__(self, nside=default_nside):
        if nside is None:
            nside = utils.set_default_nside()

        self.feature = None
        self.nside = nside

    def update_conditions(self, conditions):
        self.feature = conditions['slewtimes']
        if np.size(self.feature) > 1:
            self.feature = hp.ud_grade(self.feature, nside_out=self.nside)


class Observatory(BaseFeature):
    """Hosts informations regarding the observatory parameters.

    """
    def __init__(self, config):
        self.feature = config

class M5Depth(BaseConditionsFeature):
    """
    Given current conditions, return the 5-sigma limiting depth for a filter.
    """
    def __init__(self, filtername='r', expTime=30., nside=default_nside):
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.feature = None
        self.expTime = expTime
        self.nside = nside

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
                               conditions['FWHMeff_%s' % self.filtername][good],
                               self.expTime, conditions['airmass'][good])
        self.feature = m5
        self.feature[m5_mask] = hp.UNSEEN
        self.feature = hp.ud_grade(self.feature, nside_out=self.nside)
        self.feature = ma.masked_values(self.feature, hp.UNSEEN)


class Current_filter(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = conditions['filter']


class Mounted_filters(BaseConditionsFeature):
    def update_conditions(self, conditions):
        if 'mounted_filters' in conditions:
            self.feature = conditions['mounted_filters']
        else:
            self.feature = ['u', 'g', 'r', 'i', 'z', 'y']


class Sun_moon_alts(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = {'moonAlt': conditions['moonAlt'], 'sunAlt': conditions['sunAlt']}


class Moon(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = {'moonAlt': conditions['moonAlt'],
                        'moonAz': conditions['moonAz'],
                        'moonPhase': conditions['moonPhase'],
                        'moonRA': conditions['moonRA'],
                        'moonDec': conditions['moonDec']}


class Current_mjd(BaseConditionsFeature):
    def __init__(self):
        self.feature = -1

    def update_conditions(self, conditions):
        self.feature = conditions['mjd']


class Current_lmst(BaseConditionsFeature):
    def __init__(self):
        self.feature = -1

    def update_conditions(self, conditions):
        # XXX--mother fuck, what are the units?
        # Pretty sure this is in hours
        self.feature = conditions['lmst']


class Current_night(BaseConditionsFeature):
    def __init__(self):
        self.feature = -1

    def update_conditions(self, conditions):
        self.feature = conditions['night']

class CurrentNightBoundaries(BaseConditionsFeature):
    def __init__(self):
        self.feature = -1

    def update_conditions(self, conditions):
        self.feature = {'last_twilight_end':  conditions['last_twilight_end'],
                        'next_twilight_start': conditions['next_twilight_start']}

class Current_pointing(BaseConditionsFeature):
    def update_conditions(self, conditions):
        self.feature = {'RA': conditions['telRA'], 'dec': conditions['telDec']}


class Time_to_set(BaseConditionsFeature):
    """Map of how much time until things set.

    Warning, using very fast alt/az transformation ignoring refraction, aberration, etc.
    """
    def __init__(self, nside=default_nside, alt_min=20.):
        """
        Parameters
        ----------
        alt_min : float
            The minimum altitude one can point the telescope (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.min_alt = np.radians(alt_min)

        self.sin_dec = np.sin(self.dec)
        self.cos_dec = np.cos(self.dec)

        site = Site('LSST')
        self.sin_lat = np.sin(site.latitude_rad)
        self.cos_lat = np.cos(site.latitude_rad)
        self.lon = site.longitude_rad

        # Compute hour angle when field hits the alt_min
        ha_alt_min = -np.arccos((np.sin(self.min_alt) -
                                 self.sin_dec*self.sin_lat)/(self.cos_dec*self.cos_lat))
        self.ha_alt_min = ha_alt_min
        lmst_alt_min = ha_alt_min + self.ra
        lmst_alt_min[np.where(lmst_alt_min < 0)] += 2.*np.pi
        self.lmst_min = lmst_alt_min

        self.nans = np.isnan(self.lmst_min)

    def update_conditions(self, conditions):
        """feature = time to set in hours
        """
        lmst = conditions['lmst'] * np.pi/12.

        rad_to_limit = self.lmst_min - lmst
        rad_to_limit[np.where(rad_to_limit < 0)] += 2.*np.pi

        self.feature = rad_to_limit
        self.feature *= 12/np.pi * 365.24/366.24
        self.feature[self.nans] = hp.UNSEEN


class Time_to_alt_limit(BaseConditionsFeature):
    """Map of how much time until things set.

    Warning, using very fast alt/az transformation ignoring refraction, aberration, etc.
    """
    def __init__(self, nside=default_nside, alt_max=86.5):
        """
        Parameters
        ----------
        alt_max : float
            The maximum altitude one can point the telescope (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.max_alt = np.radians(alt_max)

        self.sin_dec = np.sin(self.dec)
        self.cos_dec = np.cos(self.dec)

        site = Site('LSST')
        self.sin_lat = np.sin(site.latitude_rad)
        self.cos_lat = np.cos(site.latitude_rad)
        self.lon = site.longitude_rad

        # compute the hour angle when a point hits the alt_max
        cos_ha = (np.sin(self.max_alt) - self.sin_dec*self.sin_lat)/(self.cos_dec * self.cos_lat)
        self.lmst_max = np.arccos(cos_ha) + self.ra
        self.nans = np.isnan(self.lmst_max)

    def update_conditions(self, conditions):
        """feature = time to set in hours
        """
        lmst = conditions['lmst'] * np.pi/12.

        rad_to_limit = self.lmst_max - lmst
        rad_to_limit[np.where(rad_to_limit < 0)] += 2.*np.pi

        self.feature = rad_to_limit
        self.feature *= 12/np.pi * 365.24/366.24
        self.feature[self.nans] = hp.UNSEEN


class Time_observable_in_night(BaseConditionsFeature):
    """
    For every healpixel, calculate the time left observable in the night
    """
    def __init__(self, nside=default_nside, max_airmass=2.5, polar_limit=-80.):
        """
        Parameters
        ----------
        max_airmass : float (2.5)
            The maximum airmass to consider a point visible
        polar_limit : float (-80.)
            Consider anything below dec polar_limit to always be visible. (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        # most fields should have a min and max lmst where they are less than max_airmass

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))

        alt_limit = np.pi/2. - np.arccos(1./max_airmass)
        site = Site('LSST')
        lat = site.latitude_rad
        self.lon = site.longitude_rad

        sinalt = np.sin(alt_limit)
        sindec = np.sin(self.dec)
        sinlat = np.sin(lat)
        cosdec = np.cos(self.dec)
        coslat = np.cos(lat)

        cosha = (sinalt - sindec*sinlat)/(cosdec*coslat)
        # Here's the hour angle (plus or minus) for each healpixel
        self.ha_limit = np.arccos(cosha)*12/np.pi

        # Consider some regions circumpolar
        self.ha_limit[np.where(self.dec < np.radians(polar_limit))] = 12.

        self.polar_limit = polar_limit

        self.feature = self.ra * 0.

    def update_conditions(self, conditions):
        """
        self.feature : healpy map
            The hours remaining for a field to be visible in the night before the
            next twilight starts.
        """
        # reset the feature value
        self.feature *= 0.

        lmst = conditions['lmst']
        # Map of the current HA, in hours
        current_ha = lmst - (self.ra*12./np.pi)
        current_ha[np.where(current_ha < 0.)] += 24
        # now to convert to -12 to 12
        over = np.where(current_ha > 12.)
        current_ha[over] = current_ha[over] - 24.

        # in hours
        time_to_twilight = (conditions['next_twilight_start'] - conditions['mjd']) * 24.

        # Check if still in twilight.
        if np.abs(conditions['mjd'] -
                  conditions['next_twilight_end']) < np.abs(conditions['mjd'] -
                                                            conditions['last_twilight_end']):
            time_left_twilight = (conditions['next_twilight_end'] - conditions['mjd']) * 24
        else:
            time_left_twilight = 0.

        # Convert from sidereal hours to regular hours. Thanks wikipedia!
        side2solar = 365.24/366.24

        # time until next setting
        self.feature = (self.ha_limit - current_ha) * side2solar

        # Crop off if hits twilight first
        self.feature[np.where(self.feature > time_to_twilight)] = time_to_twilight

        # If it needs to rise, subtract that time off
        good = np.where(current_ha < -self.ha_limit)
        time_to_rise = (-self.ha_limit[good] - current_ha[good]) * side2solar
        self.feature[good] = self.feature[good]-time_to_rise

        # If we are still in twilight, subtract that time off
        self.feature -= time_left_twilight

        # Crop off things that won't rise in time.
        self.feature[np.where(self.feature < 0)] = 0.

        # Set the polar region to be the time to twilight
        self.feature[np.where(self.dec < np.radians(self.polar_limit))] = time_to_twilight - time_left_twilight


class Rotator_angle(BaseSurveyFeature):
    """
    Track what rotation angles things are observed with.
    XXX-under construction
    """
    def __init__(self, filtername='r', binsize=10., nside=default_nside):
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

class SurveyProposals(BaseSurveyFeature):
    """
    Track the proposals which are part of this survey
    """

    def __init__(self, ids=(0,), names=('',)):
        """
        Parameters
        ----------
        filtername : string ('r')
            Filter to track.
        nside : int (32)
            Scale of the healpix map
        """

        self.id = dict(zip(ids, names))
        self.id_to_index = dict(zip(ids, range(len(ids))))
        self.feature = np.zeros(len(self.id))

    def add_observation(self, observation, indx=None):
        try:
            if observation['survey_id'][0] in self.id.keys():
                self.feature[self.id_to_index[observation['survey_id'][0]]] += 1
        except IndexError:
            if observation['survey_id'] in self.id.keys():
                self.feature[self.id_to_index[observation['survey_id']]] += 1
