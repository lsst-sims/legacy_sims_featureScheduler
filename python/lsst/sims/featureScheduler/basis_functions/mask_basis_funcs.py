import numpy as np
import healpy as hp
from lsst.sims.utils import _hpid2RaDec, Site, _angularSeparation
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.basis_functions import Base_basis_function


__all__ = ['Zenith_mask_basis_function', 'Zenith_shadow_mask_basis_function',
           'Moon_avoidance_basis_function', 'Bulk_cloud_basis_function']


class Zenith_mask_basis_function(Base_basis_function):
    """Just remove the area near zenith.

    Parameters
    ----------
    min_alt : float (20.)
        The minimum possible altitude (degrees)
    max_alt : float (82.)
        The maximum allowed altitude (degrees)
    """
    def __init__(self, min_alt=20., max_alt=82.):
        super(Zenith_mask_basis_function, self).__init__()
        self.update_on_newobs = False
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float).fill(self.penalty)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((conditions.alt > self.min_alt) &
                             (conditions.alt < self.max_alt))[0]
        result[alt_limit] = 1
        return result


class Zenith_shadow_mask_basis_function(Base_basis_function):
    """Mask the zenith, and things that will soon pass near zenith. Useful for making sure
    observations will be able to be too close to zenith when they need to be observed again (e.g. for a pair)

    Parameters
    ----------
    min_alt : float (20.)
        The minimum alititude to alow. Everything lower is masked. (degrees)
    max_alt : float (82.)
        The maximum altitude to alow. Everything higher is masked. (degrees)
    shadow_minutes : float (40.)
        Mask anything that will pass through the max alt in the next shadow_minutes time. (minutes)
    """
    def __init__(self, nside=None, min_alt=20., max_alt=82.,
                 shadow_minutes=40., penalty=np.nan, site='LSST'):
        super(Zenith_shadow_mask_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        self.penalty = penalty

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

        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(self.penalty)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((conditions.alt > self.min_alt) &
                             (conditions.alt < self.max_alt))[0]
        result[alt_limit] = 1
        to_mask = np.where((conditions.HA > (2.*np.pi-self.shadow_minutes-self.zenith_radius)) &
                           (self.decband == 1))
        result[to_mask] = np.nan
        return result


class Moon_avoidance_basis_function(Base_basis_function):
    """Avoid looking too close to the moon.

    Parameters
    ----------
    moon_distance: float (30.)
        Minimum allowed moon distance. (degrees)

    XXX--TODO:  This could be a more complicated function of filter and moon phase.
    """
    def __init__(self, nside=None, moon_distance=30.):
        super(Moon_avoidance_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        self.moon_distance = np.radians(moon_distance)
        self.result = np.ones(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        angular_distance = _angularSeparation(conditions.az, conditions.alt,
                                              conditions.moonAz,
                                              conditions.moonAlt)

        result[angular_distance < self.moon_distance] = np.nan

        return result


class Bulk_cloud_basis_function(Base_basis_function):
    """Mark healpixels on a map if their cloud values are greater than
    the same healpixels on a maximum cloud map.

    Parameters
    ----------
    nside: int (default_nside)
        The healpix resolution.
    max_cloud_map : numpy array (None)
        A healpix map showing the maximum allowed cloud values for all points on the sky
    out_of_bounds_val : float (10.)
        Point value to give regions where there are no observations requested
    """

    def __init__(self, nside=None, max_cloud_map=None, max_val=0.7,
                 out_of_bounds_val=np.nan):
        super(Bulk_cloud_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        if max_cloud_map is None:
            self.max_cloud_map = np.zeros(hp.nside2npix(nside), dtype=float) + max_val
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
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

        result = self.result.copy()

        clouded = np.where(self.max_cloud_map <= conditions.bulk_cloud)
        result[clouded] = self.out_of_bounds_val

        return result
