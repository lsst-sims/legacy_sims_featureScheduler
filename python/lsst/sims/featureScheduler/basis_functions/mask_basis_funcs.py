from __future__ import absolute_import
from builtins import object
import numpy as np
import numpy.ma as ma
import healpy as hp
from lsst.sims.featureScheduler import utils
from lsst.sims.utils import _hpid2RaDec, Site
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ['Zenith_mask_basis_function', 'Zenith_shadow_mask_basis_function']


class Zenith_mask_basis_function(Base_basis_function):
    """Just remove the area near zenith
    """
    def __init__(self, nside=None, min_alt=20., max_alt=82., penalty=0.):
        """
        """
        if nside is None:
            nside = utils.set_default_nside()
        self.penalty = penalty
        self.nside = nside
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float).fill(self.penalty)

    def __call__(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((conditions.alt > self.min_alt) &
                             (conditions.alt < self.max_alt))[0]
        result[alt_limit] = 1
        return result


class Zenith_shadow_mask_basis_function(Base_basis_function):
    """Mask the zenith, and things that will soon pass near zenith
    """
    def __init__(self, nside=None, min_alt=20., max_alt=82.,
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

        self.result = np.empty(hp.nside2npix(self.nside), dtype=float).fill(self.penalty)

    def __call__(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((conditions.alt > self.min_alt) &
                             (conditions.alt < self.max_alt))[0]
        result[alt_limit] = 1
        to_mask = np.where((conditions.HA > (2.*np.pi-self.shadow_minutes-self.zenith_radius)) &
                           (self.decband == 1))
        result[to_mask] = hp.UNSEEN
        return result
