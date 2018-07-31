from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
from . import features
from . import utils
import healpy as hp
from lsst.sims.utils import haversine, _hpid2RaDec, _angularSeparation
from lsst.sims.skybrightness_pre import M5percentiles
import matplotlib.pylab as plt
import logging

default_nside = None

log = logging.getLogger(__name__)

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

    def add_observation(self, observation, indx=None):
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)

    def update_conditions(self, conditions):
        for feature in self.condition_features:
            self.condition_features[feature].update_conditions(conditions)

    def check_feasibility(self):
        return True

    def __call__(self, **kwargs):
        """
        Return a reward healpix map or a reward scalar.
        """
        pass


class Zenith_mask_basis_function(Base_basis_function):
    """Just remove the area near zenith
    """
    def __init__(self, nside=default_nside, condition_features=None,
                 survey_features=None, min_alt=20., max_alt=82., penalty=0.):
        """
        """
        if nside is None:
            nside = utils.set_default_nside()
        self.penalty = penalty
        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)

    def __call__(self, indx=None):

        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(self.penalty)
        alt = self.condition_features['altaz'].feature['alt']
        alt_limit = np.where((alt > self.min_alt) &
                             (alt < self.max_alt))[0]
        result[alt_limit] = 1
        return result


class Quadrant_basis_function(Base_basis_function):
    """Mask regions of the sky so only certain quadrants are visible
    """
    def __init__(self, nside=default_nside, condition_features=None, minAlt=20., maxAlt=82.,
                 azWidth=15., survey_features=None, quadrants='All'):
        """
        Parameters
        ----------
        minAlt : float (20.)
            The minimum altitude to consider (degrees)
        maxAlt : float (82.)
            The maximum altitude to leave unmasked (degrees)
        azWidth : float (15.)
            The full-width azimuth to leave unmasked (degrees)
        quadrants : str ('All')
            can be 'All' or a list including any of 'N', 'E', 'S', 'W'
        """
        if nside is None:
            nside = utils.set_default_nside()

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
        # Convert to half-width for convienence
        self.azWidth = np.radians(azWidth / 2.)
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


class North_south_patch_basis_function(Base_basis_function):
    """Similar to the Quadrant_basis_function, but make it easier to
    pick up the region that passes through the zenith
    """
    def __init__(self, nside=default_nside, condition_features=None, minAlt=20., maxAlt=82.,
                 azWidth=15., survey_features=None, lat=-30.2444, zenith_pad=15., zenith_min_alt=40.):
        """
        Parameters
        ----------
        minAlt : float (20.)
            The minimum altitude to consider (degrees)
        maxAlt : float (82.)
            The maximum altitude to leave unmasked (degrees)
        azWidth : float (15.)
            The full-width azimuth to leave unmasked (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.lat = np.radians(lat)

        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)
        self.minAlt = np.radians(minAlt)
        self.maxAlt = np.radians(maxAlt)
        # Convert to half-width for convienence
        self.azWidth = np.radians(azWidth / 2.)
        self.nside = nside

        self.zenith_map = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.zenith_map.fill(hp.UNSEEN)
        hpids = np.arange(self.zenith_map.size)
        ra, dec = _hpid2RaDec(nside, hpids)
        close_dec = np.where(np.abs(dec - np.radians(lat)) < np.radians(zenith_pad))
        self.zenith_min_alt = np.radians(zenith_min_alt)
        self.zenith_map[close_dec] = 1

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        # Put in the region around the zenith
        result[np.where(self.zenith_map == 1)] = 1

        alt = self.condition_features['altaz'].feature['alt']
        az = self.condition_features['altaz'].feature['az']

        result[np.where(alt < self.zenith_min_alt)] = hp.UNSEEN
        result[np.where(alt > self.maxAlt)] = hp.UNSEEN

        result[np.where(alt > self.maxAlt)] = hp.UNSEEN
        result[np.where(alt < self.minAlt)] = hp.UNSEEN

        alt_limit = np.where((alt > self.minAlt) &
                             (alt < self.maxAlt))[0]

        q1 = np.where((az[alt_limit] > np.pi-self.azWidth) &
                      (az[alt_limit] < np.pi+self.azWidth))[0]
        result[alt_limit[q1]] = 1

        q4 = np.where((az[alt_limit] < self.azWidth) |
                      (az[alt_limit] > 2*np.pi - self.azWidth))[0]
        result[alt_limit[q4]] = 1

        return result


class HADecAltAzPatchBasisFunction(Base_basis_function):
    """Build an AltAz mask using patches defined as hour angle, declination, altitude and azimuth limits.

    """
    def __init__(self, nside=default_nside, condition_features=None, survey_features=None,
                 patches=({'ha_min': 2., 'ha_max': 22.,
                           'alt_max': 88., 'alt_min': 55.,
                           'dec_min': -90., 'dec_max': 90,
                           'az_min': 0., 'az_max': 360.,
                           'weight': 1.},)):
        """
        Parameters
        ----------
        patches : list(dict())
            A list of dictionaries containing the keywords (ha_min, ha_min), (alt_min, alt_max), (dec_min, dec_max),
            (az_min and az_max) to construct an AltAz mask. One can skip a pair of min/max value if that is not
            part of the patch definition. For instance patches = [{'alt_min': 30., 'alt_max': 86.5}] is a valid entry,
            whereas patches = [{'alt_max': 86.5}] is not.
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)
            self.condition_features['lmst'] = features.Current_lmst()

        self.patches = patches
        ra, dec = utils.ra_dec_hp_map(self.nside)
        self.ra_hours = ra*12./np.pi % 24.
        dec_deg = dec*180./np.pi

        # Pre-compute declination masks, since those won't change
        self.dec_mask = []

        for patch in self.patches:
            if 'dec_min' in patch and 'dec_max' in patch:
                dec_mask = np.bitwise_and(dec_deg >= patch['dec_min'],
                                          dec_deg <= patch['dec_max'])
            else:
                dec_mask = np.ones(hp.nside2npix(self.nside), dtype=bool)
            self.dec_mask.append(dec_mask)

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        ha = (self.condition_features['lmst'].feature - self.ra_hours) % 24.

        for i, patch in enumerate(self.patches):
            if 'ha_min' in patch and 'ha_max' in patch:
                ha_mask = np.bitwise_or(ha <= patch['ha_min'],
                                        ha >= patch['ha_max'])
            else:
                ha_mask = np.ones(hp.nside2npix(self.nside), dtype=bool)

            if 'alt_min' in patch and 'alt_max' in patch:
                alt_mask = np.bitwise_and(self.condition_features['altaz'].feature['alt'] >=
                                          patch['alt_min']*np.pi/180.,
                                          self.condition_features['altaz'].feature['alt'] <=
                                          patch['alt_max']*np.pi/180.)
            else:
                alt_mask = np.ones(hp.nside2npix(self.nside), dtype=bool)

            if 'az_min' in patch and 'az_max' in patch:
                az_mask = np.bitwise_and(self.condition_features['altaz'].feature['az'] >= patch['az_min']*np.pi/180.,
                                         self.condition_features['altaz'].feature['az'] <= patch['az_max']*np.pi/180.)
            else:
                az_mask = np.ones(hp.nside2npix(self.nside), dtype=bool)

            mask = np.bitwise_and(np.bitwise_and(np.bitwise_and(ha_mask,
                                                                alt_mask),
                                                 self.dec_mask[i]),
                                  az_mask)
            result[mask] = patch['weight'] if 'weight' in patch else 1.

        return result


class MeridianStripeBasisFunction(Base_basis_function):
    """Build an AltAz mask using a combination of meridian masks (width and height), zenith pad and a weights.

    """
    def __init__(self, nside=default_nside, condition_features=None, survey_features=None,
                 width=(15.,), height=(80,), zenith_pad=(15.,), weight=(1.,), min_alt=20., max_alt=82.):
        """
        Parameters
        ----------
        width : list (15.,)
            The width (East-West) of the meridian mask. (degrees)
        height : list (80.,)
            The height (North-South) of the meridian mask. (degrees)
        zenith_pad : list (15., )
            Radius of a circle around zenith. (degrees)
        weight : list (1.,)
            The weight for the specific region. Although any value is accepted, it should be something between [0-1.).
        min_alt : float (20.)
            Minimum allowed altitude. (degrees)
        max_alt : float (82.)
            Maximum allowed altitude. (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        len_width = len(width)
        if not all([len_width == i for i in [len(height), len(zenith_pad), len(weight)]]):
            raise ValueError('width[%i], height[%i], zenith_pad[%i] and weight[%i] must all have the same dimensions.'
                             % (len_width, len(height), len(zenith_pad), len(weight)))

        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)
            self.condition_features['lmst'] = features.Current_lmst()

        self.width = [w*np.pi/180. for w in width]  # converts to radians
        self.height = [h*np.pi/180. for h in height] # converts to radians
        self.zenith_pad_rad = [z*np.pi/180. for z in zenith_pad] # converts to radians
        self.weight = weight

        self.max_alt_rad = max_alt*np.pi/180.
        self.min_alt_rad = min_alt*np.pi/180.

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        alt = self.condition_features['altaz'].feature['alt']
        az = self.condition_features['altaz'].feature['az']

        z_dist = np.pi/2.-alt  # Zenith distance

        sin_az = np.abs(np.sin(az))
        cos_az = np.abs(np.cos(az))

        width = sin_az * z_dist
        height = cos_az * z_dist

        # Put meridian stripes. Sort by weight so larger weight goes last
        for i in np.argsort(self.weight):
            result[np.bitwise_and(width < self.width[i],
                                  height < self.height[i])] = self.weight[i]

            # Put zenith pad
            if self.zenith_pad_rad[i] > 0.:
                result[np.where(z_dist < self.zenith_pad_rad[i])] = self.weight[i]

        # Now excludes minimum and maximum altitudes
        result[np.where(alt < self.min_alt_rad)] = hp.UNSEEN
        result[np.where(alt > self.max_alt_rad)] = hp.UNSEEN

        return result


class HourAngle_bonus_basis_function(Base_basis_function):
    """Hour angle bonus basis function

    """
    def __init__(self, nside=default_nside, condition_features=None, survey_features=None,
                 max_hourangle=6., minAlt=20., maxAlt=82.):
        """
        Parameters
        ----------
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['lmst'] = features.Current_lmst()
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)

        ra, dec = utils.ra_dec_hp_map(self.nside)
        self.ra_hours = ra*12./np.pi % 24.
        self.max_hourangle = max_hourangle
        self.minAlt = np.radians(minAlt)
        self.maxAlt = np.radians(maxAlt)

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        ha = (self.condition_features['lmst'].feature - self.ra_hours) % 24.

        # Make hour angle go from -12 to 12 instead of 0 to 24
        mask = np.where(ha > 12.)
        ha[mask] -= 24.

        alt = self.condition_features['altaz'].feature['alt']

        mask = np.where((np.abs(ha) < self.max_hourangle) & (alt < self.maxAlt) & (alt > self.minAlt))
        result[mask] = (1 - np.abs(ha[mask]) / self.max_hourangle)

        return result


class NorthSouth_scan_basis_function(Base_basis_function):
    """
    """
    def __init__(self, nside=default_nside, condition_features=None, minAlt=20., maxAlt=82.,
                 length=70., survey_features=None):
        """
        Parameters
        ----------
        minAlt : float (20.)
            The minimum altitude to consider (degrees)
        maxAlt : float (82.)
            The maximum altitude to leave unmasked (degrees)
        length: float (90.)
            The lenght of the scanning region in N/S direction (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.length = np.radians(length)
        self.current_direction = 0
        self.directions = ['NS', 'SN']

        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['altaz'] = features.AltAzFeature()
            self.condition_features['current_pointing'] = features.Current_pointing()
        self.minAlt = np.radians(minAlt)
        self.maxAlt = np.radians(maxAlt)
        # Convert to half-width for convienence
        self.nside = nside

    def __call__(self, indx=None):

        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        # How far north/south the telescope is pointing
        tel_ns_axis = np.cos(self.condition_features['current_pointing'].feature['az']) * \
                      (np.radians(90.)-self.condition_features['current_pointing'].feature['alt'])

        alt = self.condition_features['altaz'].feature['alt']
        az = self.condition_features['altaz'].feature['az']

        ns_axis = np.cos(az)*(np.radians(90.)-alt)
        # If reached limit, switch direction
        if self.current_direction == 0 and tel_ns_axis > self.length:
            self.current_direction = (self.current_direction + 1) % len(self.directions)
        elif self.current_direction == 1 and tel_ns_axis < self.length:
            self.current_direction = (self.current_direction + 1) % len(self.directions)

        if self.current_direction == 0:
            result[ns_axis > tel_ns_axis] = 0.
            a = 1./(tel_ns_axis+self.length)
            b = self.length*a
            result[ns_axis <= tel_ns_axis] = (a*ns_axis[ns_axis <= tel_ns_axis] + b)
        else:
            result[ns_axis < tel_ns_axis] = 0.
            a = 1./(tel_ns_axis-self.length)
            b = -self.length*a
            result[ns_axis >= tel_ns_axis] = (a*ns_axis[ns_axis >= tel_ns_axis] + b)

        alt_limit = np.where((alt < self.minAlt) |
                             (alt > self.maxAlt))

        result[alt_limit] = hp.UNSEEN

        return result


class CableWrap_unwrap_basis_function(Base_basis_function):
    """
    """
    def __init__(self, nside=default_nside, condition_features=None, minAz=-270., maxAz=270., minAlt=20., maxAlt=82.,
                 activate_tol=20., delta_unwrap=1.2, unwrap_until=70., max_duration=30., survey_features=None):
        """
        Parameters
        ----------
        minAz : float (20.)
            The minimum azimuth to activite bf (degrees)
        maxAz : float (82.)
            The maximum azimuth to activate bf (degrees)
        unwrap_until: float (90.)
            The window in which the bf is activated (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['altaz'] = features.AltAzFeature()
            self.condition_features['current_pointing'] = features.Current_pointing()
            self.condition_features['mjd'] = features.Current_mjd()

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

    def __call__(self, indx=None):

        result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        alt = self.condition_features['altaz'].feature['alt']
        current_abs_rad = np.radians(self.condition_features['current_pointing'].feature['az'])
        unseen = np.where(np.bitwise_or(alt < self.minAlt,
                                        alt > self.maxAlt))
        result[unseen] = hp.UNSEEN

        if (self.minAz + self.activate_tol < current_abs_rad < self.maxAz - self.activate_tol) and not self.active:
            log.debug('CableWrap[Inact]telaz=%7.2f [activate@ %7.2f/%7.2f] [deactivate@ %7.2f/%7.2f]' % (
                float(np.degrees(current_abs_rad)),
                float(np.degrees(self.minAz + self.activate_tol)), float(np.degrees(self.maxAz - self.activate_tol)),
                float(np.degrees(self.minAz + self.unwrap_until)), float(np.degrees(self.maxAz - self.unwrap_until))))
            return result
        elif self.active and self.unwrap_direction == 1 and current_abs_rad > self.minAz+self.unwrap_until:
            log.debug('CableWrap[Deact:clock-wise]: telaz=%7.2f [activate@ %7.2f] [deactivate@ %7.2f]' % (
                float(np.degrees(current_abs_rad)),
                float(np.degrees(self.minAz + self.activate_tol)),
                float(np.degrees(self.minAz + self.unwrap_until))))
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result
        elif self.active and self.unwrap_direction == -1 and current_abs_rad < self.maxAz-self.unwrap_until:
            log.debug('CableWrap[Deact:counter-clock-wise]: telaz=%7.2f [activate@ %7.2f] [deactivate@ %7.2f]' % (
                float(np.degrees(current_abs_rad)),
                float(np.degrees(self.maxAz - self.activate_tol)),
                float(np.degrees(self.maxAz - self.unwrap_until))))
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result
        elif (self.activation_time is not None and
              self.condition_features['mjd'].feature - self.activation_time > self.max_duration):
            log.debug('CableWrap[Deact:timeout]: active for %5.1f min. [max %5.1f]' % (
                (self.condition_features['mjd'].feature - self.activation_time)*24.*60.,
                self.max_duration*24.*60.))
            self.active = False
            self.unwrap_direction = 0.
            self.activation_time = None
            return result

        log.debug('CableWrap[Active]: telaz=%7.2f [activate@ %7.2f/%7.2f] [deactivate@ %7.2f/%7.2f]' % (
            float(np.degrees(current_abs_rad)),
            float(np.degrees(self.minAz + self.activate_tol)), float(np.degrees(self.maxAz - self.activate_tol)),
            float(np.degrees(self.minAz + self.unwrap_until)), float(np.degrees(self.maxAz - self.unwrap_until))))

        az_rad = self.condition_features['altaz'].feature['az']

        if not self.active:
            self.activation_time = self.condition_features['mjd'].feature
            if current_abs_rad < 0.:
                self.unwrap_direction = 1  # clock-wise unwrap
            else:
                self.unwrap_direction = -1  # counter-clock-wise unwrap

        self.active = True

        max_abs_rad = self.maxAz
        min_abs_rad = self.minAz

        TWOPI = 2.*np.pi

        # Compute distance and accumulated az.
        norm_az_rad = np.divmod(az_rad - min_abs_rad, TWOPI)[1] + min_abs_rad
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


class Target_map_basis_function(Base_basis_function):
    """Normalize the maps first to make things smoother
    """
    def __init__(self, filtername='r', nside=default_nside, target_map=None,
                 survey_features=None, condition_features=None, norm_factor=240./2.5e6,
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
        norm_factor : float (800./2.5e6)
            for converting target map to number of observations. This is a convience scaling,
            It should be degenerate with the weight of the basis function.
        out_of_bounds_val : float (10.)
            Point value to give regions where there are no observations requested
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.norm_factor = norm_factor
        if survey_features is None:
            self.survey_features = {}
            # Map of the number of observations in filter
            self.survey_features['N_obs'] = features.N_observations(filtername=filtername)
            # Count of all the observations
            self.survey_features['N_obs_count_all'] = features.N_obs_count(filtername=None)
        super(Target_map_basis_function, self).__init__(survey_features=self.survey_features,
                                                        condition_features=condition_features)
        self.nside = nside
        if target_map is None:
            self.target_map = utils.generate_goal_map(filtername=filtername)
        else:
            self.target_map = target_map
        self.out_of_bounds_area = np.where(self.target_map == 0)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.inside_area = np.where(self.target_map != 0)

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

        n_max = np.max(self.survey_features['N_obs'].feature[self.inside_area])
        n_min = np.min(self.survey_features['N_obs'].feature[self.inside_area])

        if n_max > 0:
            goal = (n_max - self.survey_features['N_obs'].feature) / (n_max - n_min)
            result[self.inside_area] += self.target_map[self.inside_area] * goal[self.inside_area]
        else:
            result[self.inside_area] += self.target_map[self.inside_area]

        # # Find out how many observations we want now at those points
        # goal_N = self.target_map[indx] * self.survey_features['N_obs_count_all'].feature * self.norm_factor
        #
        # result[indx] = goal_N - self.survey_features['N_obs'].feature[indx]
        #
        # result[self.out_of_bounds_area] = self.out_of_bounds_val
        # norm_val = np.max(result[self.inside_area])v
        # result[self.inside_area] -= (norm_val-1.)

        return result[indx]


class Avoid_Fast_Revists(Base_basis_function):
    """Marks targets as unseen if they are in a specified time window in order to avoid fast revisits.
    """
    def __init__(self, filtername='r', nside=default_nside, gap_min=25.,
                 survey_features=None, condition_features=None,
                 out_of_bounds_val=-10.):
        """
        Parameters
        ----------
        filtername: (string 'r')
            The name of the filter for this target map.
        gap_min : float (25.)
            Minimum time for the gap (minutes).
        nside: int (default_nside)
            The healpix resolution.
        survey_features:
        condition_features:
        out_of_bounds_val: float (10.)
            Point value to give regions where there are no observations requested
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.gap_min = gap_min/60./24.
        self.nside = nside
        self.out_of_bounds_val = out_of_bounds_val

        if survey_features is None:
            self.survey_features = dict()
            self.survey_features['Last_observed'] = features.Last_observed(filtername=filtername)

        if condition_features is None:
            self.condition_features = {}
            # Current MJD
            self.condition_features['Current_mjd'] = features.Current_mjd()

        super(Avoid_Fast_Revists, self).__init__(survey_features=self.survey_features,
                                                 condition_features=self.condition_features)

    def __call__(self, indx=None):
        result = np.ones(hp.nside2npix(self.nside), dtype=float)
        if indx is None:
            indx = np.arange(result.size)
        diff = self.condition_features['Current_mjd'].feature - self.survey_features['Last_observed'].feature[indx]
        bad = np.where(diff < self.gap_min)[0]
        result[indx[bad]] = hp.UNSEEN
        return result


class Visit_repeat_basis_function(Base_basis_function):
    """
    Basis function to reward re-visiting an area on the sky. Looking for Solar System objects.
    """
    def __init__(self, survey_features=None, condition_features=None, gap_min=25., gap_max=45.,
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
        if nside is None:
            nside = utils.set_default_nside()

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


class M5_diff_basis_function(Base_basis_function):
    """Basis function based on the 5-sigma depth.
    Look up the best a pixel gets, and compute the limiting depth difference with current conditions
    """
    def __init__(self, survey_features=None, condition_features=None, filtername='r',
                 nside=default_nside):
        """
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.nside = nside

        # Need to look up the deepest m5 values for all the healpixels
        m5p = M5percentiles()
        self.dark_map = m5p.dark_map(filtername=filtername, nside_out=self.nside)
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['M5Depth'] = features.M5Depth(filtername=filtername, nside=nside)
        super(M5_diff_basis_function, self).__init__(survey_features=survey_features,
                                                     condition_features=self.condition_features)

    def __call__(self, indx=None):
        # No way to get the sign on this right the first time.
        result = self.condition_features['M5Depth'].feature - self.dark_map
        mask = np.where(self.condition_features['M5Depth'].feature.filled() == hp.UNSEEN)
        result[mask] = hp.UNSEEN
        return result


class Teff_basis_function(Base_basis_function):
    """Basis function based on the effective exposure time.
    Look up the faintest a pixel gets, and compute the teff difference with current conditions
    """
    def __init__(self, survey_features=None, condition_features=None, filtername='r', nside=default_nside,
                 texp=30.):
        """
        Parameters
        ----------
        texp : float (30.)
            The exposure time to scale to (seconds).
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.filtername = filtername
        self.nside = nside
        self.texp = texp

        # Need to look up the deepest m5 values for all the healpixels
        m5p = M5percentiles()
        self.dark_map = m5p.dark_map(filtername=filtername, nside_out=self.nside)
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['M5Depth'] = features.M5Depth(filtername=filtername, nside=nside)
        super(Teff_basis_function, self).__init__(survey_features=survey_features,
                                                  condition_features=self.condition_features)

    def __call__(self, indx=None):
        # No way to get the sign on this right the first time.
        mag_diff = self.condition_features['M5Depth'].feature - self.dark_map
        mask = np.where(self.condition_features['M5Depth'].feature.filled() == hp.UNSEEN)
        result = 10.**(0.8*mag_diff)*self.texp
        result[mask] = hp.UNSEEN
        return result


class Strict_filter_basis_function(Base_basis_function):
    """Remove the bonus for staying in the same filter if certain conditions are met.

    If the moon rises/sets or twilight starts/ends, it makes a lot of sense to consider
    a filter change. This basis function rewards if it matches the current filter, the moon rises or sets,
    twilight starts or stops, or there has been a large gap since the last observation.

    """
    def __init__(self, survey_features=None, condition_features=None, time_lag_min=10., time_lag_max=30.,
                 time_lag_boost=60., boost_gain=2.0, unseen_before_lag=False,
                 filtername='r', tag=None, twi_change=-18., proportion=1.0, aways_available=False):
        """
        Paramters
        ---------
        time_lag : float (10.)
            If there is a gap between observations longer than this, let the filter change (minutes)
        twi_change : float (-18.)
            The sun altitude to consider twilight starting/ending
        """
        self.time_lag_min = time_lag_min/60./24.  # Convert to days
        self.time_lag_max = time_lag_max / 60. / 24.  # Convert to days
        self.time_lag_boost = time_lag_boost / 60. / 24.
        self.boost_gain = boost_gain
        self.unseen_before_lag = unseen_before_lag

        self.twi_change = np.radians(twi_change)
        self.filtername = filtername
        self.proportion = proportion
        self.aways_available = aways_available

        if condition_features is None:
            self.condition_features = {}
            self.condition_features['Current_filter'] = features.Current_filter()
            self.condition_features['Mounted_filter'] = features.Mounted_filters()
            self.condition_features['Sun_moon_alts'] = features.Sun_moon_alts()
            self.condition_features['Current_mjd'] = features.Current_mjd()
        if survey_features is None:
            self.survey_features = {}
            self.survey_features['Last_observation'] = features.Last_observation()
            self.survey_features['N_obs_all'] = features.N_obs_count(filtername=None)
            self.survey_features['N_obs'] = features.N_obs_count(filtername=filtername,
                                                                 tag=tag)

        super(Strict_filter_basis_function, self).__init__(survey_features=self.survey_features,
                                                           condition_features=self.condition_features)

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
        need = 1.-nobs/nobs_all+goal if nobs_all > 0 else 1.+goal
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

        return bonus*need

    def check_feasibility(self):
        if self.condition_features['Current_filter'].feature is None or \
                self.condition_features['Current_filter'].feature == self.filtername or self.aways_available:
            return True

        # Did the moon set or rise since last observation?
        moon_changed = self.condition_features['Sun_moon_alts'].feature['moonAlt'] * \
                       self.survey_features['Last_observation'].feature['moonAlt'] < 0

        # Are we already in the filter (or at start of night)?
        not_in_filter = (self.condition_features['Current_filter'].feature != self.filtername)

        # Has enough time past?
        lag = self.condition_features['Current_mjd'].feature - self.survey_features['Last_observation'].feature['mjd']
        time_past = lag > self.time_lag_min

        # Did twilight start/end?
        twi_changed = (self.condition_features['Sun_moon_alts'].feature['sunAlt'] - self.twi_change) * \
                      (self.survey_features['Last_observation'].feature['sunAlt'] - self.twi_change) < 0

        # Did we just finish a DD sequence
        wasDD = self.survey_features['Last_observation'].feature['note'] == 'DD'

        # Is the filter mounted?
        mounted = self.filtername in self.condition_features['Mounted_filter'].feature

        if (moon_changed | time_past | twi_changed | wasDD) & mounted & not_in_filter:
            return True
        else:
            return False

    def __call__(self, **kwargs):

        if self.condition_features['Current_filter'].feature is None:
            return 0.  # no bonus if no filter is mounted
        elif self.condition_features['Current_filter'].feature == self.filtername:
            return 0.  # no bonus if on the filter already

        # Did the moon set or rise since last observation?
        moon_changed = self.condition_features['Sun_moon_alts'].feature['moonAlt'] * self.survey_features['Last_observation'].feature['moonAlt'] < 0

        # Are we already in the filter (or at start of night)?
        not_in_filter = (self.condition_features['Current_filter'].feature != self.filtername)

        # Has enough time past?
        lag = self.condition_features['Current_mjd'].feature - self.survey_features['Last_observation'].feature['mjd']
        time_past = lag > self.time_lag_min

        # Did twilight start/end?
        twi_changed = (self.condition_features['Sun_moon_alts'].feature['sunAlt'] - self.twi_change) * (self.survey_features['Last_observation'].feature['sunAlt']- self.twi_change) < 0

        # Did we just finish a DD sequence
        wasDD = self.survey_features['Last_observation'].feature['note'] == 'DD'

        # Is the filter mounted?
        mounted = self.filtername in self.condition_features['Mounted_filter'].feature

        if (moon_changed | time_past | twi_changed | wasDD) & mounted & not_in_filter:
            # if we are here and time has not passed, give zero bonus (instead of -np.inf in case of unseen_before_lag)
            if lag > self.time_lag_boost:
                log.debug('Filter change boost active! %s[%s]: %f' % (self.filtername,
                                                                      self.condition_features['Current_filter'].feature,
                                                                      self.filter_change_bonus(lag)))
            result = self.filter_change_bonus(lag) if time_past else 0.
        else:
            result = -100. if self.unseen_before_lag else 0.

        return result


class Rolling_mask_basis_function(Base_basis_function):
    """Have a simple mask that turns on and off
    """
    def __init__(self, mask=None, year_mod=2, year_offset=0,
                 survey_features=None, condition_features=None,
                 nside=default_nside, mjd_start=0):
        """
        Parameters
        ----------
        mask : array (bool)
            A HEALpix map that marks which pixels should be masked on even years
        mjd_start : float
            The starting MJD of the survey (days)
        year_mod : int (2)
            How often should the mask be toggled on
        year_offset : int (0)
            A possible offset to when the mask starts/stops
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.mjd_start = mjd_start
        self.mask = np.where(mask == True)
        self.year_mod = year_mod
        self.year_offset = year_offset
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['mjd'] = features.Current_mjd()
        super(Rolling_mask_basis_function, self).__init__(survey_features=survey_features,
                                                          condition_features=self.condition_features)
        self.nomask = np.ones(hp.nside2npix(nside))

    def __call__(self, **kwargs):
        """If year is even, apply mask, otherwise, not
        """
        year = np.floor((self.condition_features['mjd'].feature-self.mjd_start)/365.25)
        if (year + self.year_offset) % self.year_mod == 0:
            # This is a year we should turn the mask on
            result = self.nomask.copy()
            result[self.mask] = hp.UNSEEN
        else:
            # Not a mask year, all pixels are live
            result = self.nomask
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
                 max_time=135., order=1., hard_max=None, filtername='r', nside=default_nside):
        if nside is None:
            nside = utils.set_default_nside()

        self.maxtime = max_time
        self.hard_max = hard_max
        self.order = order
        self.nside = nside
        self.filtername = filtername
        if condition_features is None:
            self.condition_features = {}
            self.condition_features['Current_filter'] = features.Current_filter()
            self.condition_features['hp2fields'] = features.HP2Fields()
            self.condition_features['slewtime'] = features.SlewtimeFeature(nside=nside)
        super(Slewtime_basis_function, self).__init__(survey_features=survey_features,
                                                      condition_features=self.condition_features)

    def __call__(self, indx=None):
        # If we are in a different filter, the Filter_change_basis_function will take it
        if self.condition_features['Current_filter'].feature != self.filtername:
            result = 0.
        else:
            # Need to make sure smaller slewtime is larger reward.
            if np.size(self.condition_features['slewtime'].feature) > 1:
                result = np.empty(np.size(self.condition_features['slewtime'].feature), dtype=float)
                result.fill(hp.UNSEEN)

                good = np.where(np.bitwise_and(self.condition_features['slewtime'].feature > 0.,
                                               self.condition_features['slewtime'].feature < self.maxtime))
                result[good] = ((self.maxtime - self.condition_features['slewtime'].feature[good]) /
                                self.maxtime)**self.order
                if self.hard_max is not None:
                    not_so_good = np.where(self.condition_features['slewtime'].feature > self.hard_max)
                    result[not_so_good] -= 10.
                fields = np.unique(self.condition_features['hp2fields'].feature[good])
                for field in fields:
                    hp_indx = np.where(self.condition_features['hp2fields'].feature == field)
                    result[hp_indx] = np.min(result[hp_indx])
            else:
                result = (self.maxtime - self.condition_features['slewtime'].feature)/self.maxtime
        return result


class Bulk_cloud_basis_function(Base_basis_function):
    """Mark healpixels on a map as unseen if their cloud values are greater than
    the same healpixels on a maximum cloud map.

    """

    def __init__(self, nside=default_nside, max_cloud_map=None,
                 survey_features=None, condition_features=None, out_of_bounds_val=-10.):
        """
        Parameters
        ----------
        nside: int (default_nside)
            The healpix resolution.
        max_cloud_map : numpy array (None)
            A healpix map showing the maximum allowed cloud values for all points on the sky
        survey_features : dict, opt
        condition_features : dict, opt
        out_of_bounds_val : float (10.)
            Point value to give regions where there are no observations requested
        """
        if nside is None:
            nside = utils.set_default_nside()

        if survey_features is None:
            self.survey_features = dict()
        else:
            self.survey_features = survey_features

        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['bulk_cloud'] = features.BulkCloudCover()
        else:
            self.condition_features = condition_features

        super(Bulk_cloud_basis_function, self).__init__(survey_features=self.survey_features,
                                                        condition_features=self.condition_features)
        self.nside = nside
        if max_cloud_map is None:
            self.max_cloud_map = np.ones(hp.nside2npix(nside), dtype=float)
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.)[0]
        self.out_of_bounds_val = out_of_bounds_val

    def __call__(self, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix map where pixels with a cloud value greater than the max_cloud_map
        value are marked as unseen.
        """

        result = np.ones(hp.nside2npix(self.nside))

        clouded = np.where(self.max_cloud_map < self.condition_features['bulk_cloud'].feature)
        result[clouded] = hp.UNSEEN

        return result


class Moon_avoidance_basis_function(Base_basis_function):
    """Mark regions that are closer than a certain .

    """
    def __init__(self, nside=default_nside, condition_features=None, survey_features=None,
                 moon_distance=30.):
        """
        Parameters
        moon_distance: float (30.)
            Minimum allowed moon distance. (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['altaz'] = features.AltAzFeature(nside=nside)
            self.condition_features['lmst'] = features.Current_lmst()
            self.condition_features['moon'] = features.Moon()

        self.moon_distance = np.radians(moon_distance)

    def __call__(self, indx=None):
        result = np.ones(hp.nside2npix(self.nside), dtype=float)

        alt = self.condition_features['altaz'].feature['alt']
        az = self.condition_features['altaz'].feature['az']

        angular_distance = _angularSeparation(az, alt,
                                              self.condition_features['moon'].feature['moonAz'],
                                              self.condition_features['moon'].feature['moonAlt'])

        result[angular_distance < self.moon_distance] = hp.UNSEEN

        return result


class Skybrightness_limit_basis_function(Base_basis_function):
    """Mark regions that are closer than a certain .

    """
    def __init__(self, nside=default_nside, condition_features=None, survey_features=None,
                 filtername='r', min=20., max=30.):
        """
        Parameters
        moon_distance: float (30.)
            Minimum allowed moon distance. (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.min = min
        self.max = max
        self.nside = nside
        if survey_features is None:
            self.survey_features = {}
        if condition_features is None:
            self.condition_features = dict()
            self.condition_features['skybrightness'] = features.SkyBrightness(filtername=filtername, nside=nside)

    def __call__(self, indx=None):
        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(hp.UNSEEN)

        good = np.where(np.bitwise_and(self.condition_features['skybrightness'].feature > self.min,
                                       self.condition_features['skybrightness'].feature < self.max))
        result[good] = 1.0

        return result