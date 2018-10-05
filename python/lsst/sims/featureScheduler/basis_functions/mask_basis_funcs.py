from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
from . import features
from . import utils
import healpy as hp
from lsst.sims.utils import _hpid2RaDec, Site, _angularSeparation
from lsst.sims.skybrightness_pre import M5percentiles
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function




class Zenith_mask_basis_function(Base_basis_function):
    """Just remove the area near zenith
    """
    def __init__(self, nside=None, condition_features=None,
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


class Zenith_shadow_mask_basis_function(Base_basis_function):
    """Mask the zenith, and things that will soon pass near zenith
    """
    def __init__(self, nside=None, condition_features=None,
                 survey_features=None, min_alt=20., max_alt=82.,
                 shadow_minutes=40., penalty=hp.UNSEEN, site='LSST'):
        """
        Parameters
        ----------
        min_alt : float (20.)
            The minimum alititude to alow. Everything lower is masked. (degrees)
        max_alt : float (82.)
            The maximum altitude to alow. Everything higher is masked. (degrees)
        shadow_minutes : float (40.)
            Mask anything that will pass through the max alt in the next shadow_minutes time. (minutes)
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
            self.condition_features['lmst'] = features.Current_lmst()
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.shadow_minutes = np.radians(shadow_minutes/60. * 360./24.)
        # Compute the declination band where things could drift into zenith
        self.decband = np.zeros(self.dec.size, dtype=float)
        self.zenith_radius = np.radians(90.-max_alt)/2.
        site = Site(name=site)
        self.lat_rad = site.latitude_rad
        self.lon_rad = site.longitude_rad
        self.decband[np.where((self.dec < (self.lat_rad+self.zenith_radius)) &
                              (self.dec > (self.lat_rad-self.zenith_radius)))] = 1

    def __call__(self, indx=None):

        result = np.empty(hp.nside2npix(self.nside), dtype=float)
        result.fill(self.penalty)
        alt = self.condition_features['altaz'].feature['alt']
        alt_limit = np.where((alt > self.min_alt) &
                             (alt < self.max_alt))[0]
        result[alt_limit] = 1
        HA = np.radians(self.condition_features['lmst'].feature*360./24.) - self.ra
        HA[np.where(HA < 0)] += 2.*np.pi
        to_mask = np.where((HA > (2.*np.pi-self.shadow_minutes-self.zenith_radius)) & (self.decband == 1))
        result[to_mask] = hp.UNSEEN
        return result


class Quadrant_basis_function(Base_basis_function):
    """Mask regions of the sky so only certain quadrants are visible
    """
    def __init__(self, nside=None, condition_features=None, minAlt=20., maxAlt=82.,
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
    def __init__(self, nside=None, condition_features=None, minAlt=20., maxAlt=82.,
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
    def __init__(self, nside=None, condition_features=None, survey_features=None,
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
    def __init__(self, nside=None, condition_features=None, survey_features=None,
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

class Rolling_mask_basis_function(Base_basis_function):
    """Have a simple mask that turns on and off
    """
    def __init__(self, mask=None, year_mod=2, year_offset=0,
                 survey_features=None, condition_features=None,
                 nside=None, mjd_start=0):
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
