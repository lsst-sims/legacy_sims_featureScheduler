"""Footprints: Some relevant LSST footprints, including utilities to build them.

The goal here is to make it easy to build typical target maps and then their associated combined
survey inputs (maps in each filter, including scaling between filters; the associated cloud and
sky brightness maps that would have limits for WFD, etc.).

For generic use for defining footprints from scratch, there is also a utility that simply generates
the healpix points across the sky, along with their corresponding RA/Dec/Galactic l,b/Ecliptic l,b values.
"""

import os
import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from .utils import set_default_nside, int_rounded
from lsst.sims.utils import _hpid2RaDec, _angularSeparation
from lsst.sims.utils import Site
from lsst.utils import getPackageDir

__all__ = ['ra_dec_hp_map', 'generate_all_sky', 'get_dustmap',
           'WFD_healpixels', 'WFD_no_gp_healpixels', 'WFD_bigsky_healpixels', 'WFD_no_dust_healpixels',
           'SCP_healpixels', 'NES_healpixels',
           'galactic_plane_healpixels', #'low_lat_plane_healpixels', 'bulge_healpixels',
           'magellanic_clouds_healpixels',
           'generate_goal_map', 'standard_goals',
           'calc_norm_factor', 'filter_count_ratios']


def ra_dec_hp_map(nside=None):
    """
    Return all the RA,dec points for the centers of a healpix map, in radians.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec


def get_dustmap(nside=None):
    if nside is None:
        nside = set_default_nside()
    ebvDataDir = getPackageDir('sims_maps')
    filename = 'DustMaps/dust_nside_%i.npz' % nside
    dustmap = np.load(os.path.join(ebvDataDir, filename))['ebvMap']
    return dustmap


def generate_all_sky(nside=None, elevation_limit=20, mask=hp.UNSEEN):
    """Set up a healpix map over the entire sky.
    Calculate RA & Dec, Galactic l & b, Ecliptic l & b, for all healpixels.
    Calculate max altitude, to set to  areas which LSST cannot reach (set these to hp.unseen).

    This is intended to be a useful tool to use to set up target maps, beyond the standard maps
    provided in these utilities. Masking based on RA, Dec, Galactic or Ecliptic lat and lon is easier.

    Parameters
    ----------
    nside : int, opt
        Resolution for the healpix maps.
        Default None uses lsst.sims.featureScheduler.utils.set_default_nside to set default (often 32).
    elevation_limit : float, opt
        Elevation limit for map.
        Parts of the sky which do not reach this elevation limit will be set to mask.
    mask : float, opt
        Mask value for 'unreachable' parts of the sky, defined as elevation < 20.
        Note that the actual limits will be set elsewhere, using the observatory model.
        This limit is for use when understanding what the maps could look like.

    Returns
    -------
    dict of np.ndarray
        Returns map, RA/Dec, Gal l/b, Ecl l/b (each an np.ndarray IN RADIANS) in a dictionary.
    """
    if nside is None:
        nside = set_default_nside()

    # Calculate coordinates of everything.
    skymap = np.zeros(hp.nside2npix(nside), float)
    ra, dec = ra_dec_hp_map(nside=nside)
    coord = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame='icrs')
    eclip_lat = coord.barycentrictrueecliptic.lat.deg
    eclip_lon = coord.barycentrictrueecliptic.lon.deg
    gal_lon = coord.galactic.l.deg
    gal_lat = coord.galactic.b.deg

    # Calculate max altitude (when on meridian).
    lsst_site = Site('LSST')
    elev_max = np.pi / 2. - np.abs(dec - lsst_site.latitude_rad)
    skymap = np.where(int_rounded(elev_max) >= int_rounded(np.radians(elevation_limit), skymap, mask))

    return {'map': skymap, 'ra': np.degrees(ra), 'dec': np.degrees(dec),
            'eclip_lat': eclip_lat, 'eclip_lon': eclip_lon,
            'gal_lat': gal_lat, 'gal_lon': gal_lon}


def WFD_healpixels(nside=None, dec_min=-62.5, dec_max=3.6):
    """
    Define a region based on declination limits only.

    Parameters
    ----------
    nside : int, opt
        Resolution for the healpix maps.
        Default None uses lsst.sims.featureScheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, opt
        Minimum declination of the region (deg). Default -62.5.
    dec_max : float, opt
        Maximum declination of the region (deg). Default 3.6.

    Returns
    -------
    np.ndarray
        Healpix map with regions in declination-limited 'wfd' region as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    dec = int_rounded(dec)
    good = np.where((dec >= int_rounded(np.radians(dec_min))) &
                    (dec <= int_rounded(np.radians(dec_max))))
    result[good] = 1
    return result


def WFD_no_gp_healpixels(nside, dec_min=-62.5, dec_max=3.6,
                         center_width=10., end_width=4., gal_long1=290., gal_long2=70.):
    """
    Define a wide fast deep region with a galactic plane limit.

    Parameters
    ----------
    nside : int, opt
        Resolution for the healpix maps.
        Default None uses lsst.sims.featureScheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, opt
        Minimum declination of the region (deg).
    dec_max : float, opt
        Maximum declination of the region (deg).
    center_width : float, opt
        Width across the central part of the galactic plane region.
    end_width : float, opt
        Width across the remainder of the galactic plane region.
    gal_long1 : float, opt
        Longitude at which to start tapering from center_width to end_width.
    gal_long2 : float, opt
        Longitude at which to stop tapering from center_width to end_width.

    Returns
    -------
    np.ndarray
        Healpix map with regions in declination-limited 'wfd' region as 1.
    """
    wfd_dec = WFD_healpixels(nside, dec_min=dec_min, dec_max=dec_max)
    gp = galactic_plane_healpixels(nside=nside, center_width=center_width, end_width=end_width,
                                   gal_long1=gal_long1, gal_long2=gal_long2)
    sky = np.where(wfd_dec - gp > 0, wfd_dec - gp, 0)
    return sky


def WFD_bigsky_healpixels(nside):
    sky = WFD_no_gp_healpixels(nside, dec_min=-72.25, dec_max=12.4, center_width=14.9,
                               gal_long1=0, gal_long2=360)
    return sky


def WFD_no_dust_healpixels(nside, dec_min=-72.25, dec_max=12.4, dust_limit=0.19):
    """Define a WFD region with a dust extinction limit.

    Parameters
    ----------
    nside : int, opt
        Resolution for the healpix maps.
        Default None uses lsst.sims.featureScheduler.utils.set_default_nside to set default (often 32).
    dec_min : float, opt
        Minimum dec of the region (deg). Default -72.5 deg.
    dec_max : float, opt.
        Maximum dec of the region (deg). Default 12.5 deg.
        1.75 is the FOV radius in deg.
    dust_limit : float, None
        Remove pixels with E(B-V) values greater than dust_limit from the footprint.

    Returns
    -------
    result : numpy array
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    dustmap = get_dustmap(nside)

    result = np.zeros(ra.size, float)
    # First set based on dec range.
    dec = int_rounded(dec)
    good = np.where((dec >= int_rounded(np.radians(dec_min))) &
                    (dec <= int_rounded(np.radians(dec_max))))
    result[good] = 1
    # Now remove areas with dust extinction beyond the limit.
    result = np.where(dustmap >= dust_limit, 0, result)
    return result


def SCP_healpixels(nside=None, dec_max=-60.):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    good = np.where(int_rounded(dec) < int_rounded(np.radians(dec_max)))
    result[good] = 1
    return result


def NES_healpixels(nside=None, min_EB=-30.0, max_EB = 10.0, dec_min=2.8):
    """
    Define the North Ecliptic Spur region. Return a healpix map with NES pixels as 1.

    Parameters
    ----------
    nside : int
        A valid healpix nside
    min_EB : float (-30.)
        Minimum barycentric true ecliptic latitude (deg)
    max_EB : float (10.)
        Maximum barycentric true ecliptic latitude (deg)
    dec_min : float (2.8)
        Minimum dec in degrees

    Returns
    -------
    result : numpy array
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size, float)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    eclip_lat = int_rounded(eclip_lat)
    dec = int_rounded(dec)
    good = np.where((eclip_lat > int_rounded(np.radians(min_EB))) &
                    (eclip_lat < int_rounded(np.radians(max_EB))) &
                    (dec > int_rounded(np.radians(dec_min))))
    result[good] = 1

    return result


def galactic_plane_healpixels(nside=None, center_width=10., end_width=4.,
                              gal_long1=290., gal_long2=70.):
    """
    Define a Galactic Plane region.

    Parameters
    ----------
    nside : int, opt
        Resolution for the healpix maps.
        Default None uses lsst.sims.featureScheduler.utils.set_default_nside to set default (often 32).
    center_width : float, opt
        Width at the center of the galactic plane region.
    end_width : float, opt
        Width at the remainder of the galactic plane region.
    gal_long1 : float, opt
        Longitude at which to start the GP region.
    gal_long2 : float, opt
        Longitude at which to stop the GP region.
        Order matters for gal_long1 / gal_long2!

    Returns
    -------
    np.ndarray
        Healpix map with galactic plane regions set to 1.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = ra_dec_hp_map(nside=nside)

    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    gal_lon, gal_lat = coord.galactic.l.deg, coord.galactic.b.deg
    # Reject anything beyond the central width.
    sky = np.where(np.abs(gal_lat) < center_width, 1, 0)
    # Apply the galactic longitude cuts, so that plane goes between gal_long1 to gal_long2.
    # This is NOT the shortest distance between the angles.
    gp_length = (gal_long2 - gal_long1) % 360
    # If the length is greater than 0 then we can add additional cuts.
    if gp_length > 0:
        # First, remove anything outside the gal_long1/gal_long2 region.
        sky = np.where(int_rounded((gal_lon - gal_long1) % 360) < int_rounded(gp_length), sky, 0)
        # Add the tapers.
        # These slope from the center (gp_center @ center_width)
        # to the edge (gp_center + gp_length/2 @ end_width).
        half_width = gp_length / 2.
        slope = (center_width - end_width) / half_width
        gp_center = (gal_long1 + half_width) % 360
        gp_dist = gal_lon - gp_center
        gp_dist = np.abs(np.where(int_rounded(gp_dist) > int_rounded(180), (180 - gp_dist) % 180, gp_dist))
        lat_limit = np.abs(center_width - slope * gp_dist)
        sky = np.where(int_rounded(np.abs(gal_lat)) < int_rounded(lat_limit), sky, 0)
    return sky


def magellanic_clouds_healpixels(nside=None, lmc_radius=10, smc_radius=5):
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(hp.nside2npix(nside))

    lmc_ra = np.radians(80.893860)
    lmc_dec = np.radians(-69.756126)
    lmc_radius = np.radians(lmc_radius)

    smc_ra = np.radians(13.186588)
    smc_dec = np.radians(-72.828599)
    smc_radius = np.radians(smc_radius)

    dist_to_lmc = _angularSeparation(lmc_ra, lmc_dec, ra, dec)
    lmc_pix = np.where(int_rounded(dist_to_lmc) < int_rounded(lmc_radius))
    result[lmc_pix] = 1

    dist_to_smc = _angularSeparation(smc_ra, smc_dec, ra, dec)
    smc_pix = np.where(int_rounded(dist_to_smc) < int_rounded(smc_radius))
    result[smc_pix] = 1
    return result


def generate_goal_map(nside=None, NES_fraction = .3, WFD_fraction = 1.,
                      SCP_fraction=0.4, GP_fraction = 0.2,
                      NES_min_EB = -30., NES_max_EB = 10, NES_dec_min = 3.6,
                      SCP_dec_max=-62.5, gp_center_width=10.,
                      gp_end_width=4., gp_long1=290., gp_long2=70.,
                      wfd_dec_min=-62.5, wfd_dec_max=3.6,
                      generate_id_map=False):
    """
    Handy function that will put together a target map in the proper order.
    """
    if nside is None:
        nside = set_default_nside()

    # Note, some regions overlap, thus order regions are added is important.
    result = np.zeros(hp.nside2npix(nside), dtype=float)
    id_map = np.zeros(hp.nside2npix(nside), dtype=int)
    pid = 1
    prop_name_dict = dict()

    if NES_fraction > 0.:
        nes = NES_healpixels(nside=nside, min_EB = NES_min_EB, max_EB = NES_max_EB,
                             dec_min=NES_dec_min)
        result[np.where(nes != 0)] = 0
        result += NES_fraction*nes
        id_map[np.where(nes != 0)] = 1
        pid += 1
        prop_name_dict[1] = 'NorthEclipticSpur'

    if WFD_fraction > 0.:
        wfd = WFD_healpixels(nside=nside, dec_min=wfd_dec_min, dec_max=wfd_dec_max)
        result[np.where(wfd != 0)] = 0
        result += WFD_fraction*wfd
        id_map[np.where(wfd != 0)] = 3
        pid += 1
        prop_name_dict[3] = 'WideFastDeep'

    if SCP_fraction > 0.:
        scp = SCP_healpixels(nside=nside, dec_max=SCP_dec_max)
        result[np.where(scp != 0)] = 0
        result += SCP_fraction*scp
        id_map[np.where(scp != 0)] = 2
        pid += 1
        prop_name_dict[2] = 'SouthCelestialPole'

    if GP_fraction > 0.:
        gp = galactic_plane_healpixels(nside=nside, center_width=gp_center_width,
                                       end_width=gp_end_width, gal_long1=gp_long1,
                                       gal_long2=gp_long2)
        result[np.where(gp != 0)] = 0
        result += GP_fraction*gp
        id_map[np.where(gp != 0)] = 4
        pid += 1
        prop_name_dict[4] = 'GalacticPlane'

    if generate_id_map:
        return result, id_map, prop_name_dict
    else:
        return result


def standard_goals(nside=None):
    """
    A quick function to generate the "standard" goal maps. This is the traditional WFD/mini survey footprint.
    """
    if nside is None:
        nside = set_default_nside()

    result = {}
    result['u'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.31, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    result['g'] = generate_goal_map(nside=nside, NES_fraction=0.2,
                                    WFD_fraction=0.44, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    result['r'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    result['i'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    result['z'] = generate_goal_map(nside=nside, NES_fraction=0.4,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    result['y'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15,
                                    wfd_dec_min=-62.5, wfd_dec_max=3.6)
    return result


def calc_norm_factor(goal_dict, radius=1.75):
    """Calculate how to normalize a Target_map_basis_function.
    This is basically:
    the area of the fov / area of a healpixel  /
    the sum of all of the weighted-healpix values in the footprint.

    Parameters
    -----------
    goal_dict : dict of healpy maps
        The target goal map(s) being used
    radius : float (1.75)
        Radius of the FoV (degrees)

    Returns
    -------
    Value to use as Target_map_basis_function norm_factor kwarg
    """
    all_maps_sum = 0
    for key in goal_dict:
        good = np.where(goal_dict[key] > 0)
        all_maps_sum += goal_dict[key][good].sum()
    nside = hp.npix2nside(goal_dict[key].size)
    hp_area = hp.nside2pixarea(nside, degrees=True)
    norm_val = radius**2*np.pi/hp_area/all_maps_sum
    return norm_val


def filter_count_ratios(target_maps):
    """Given the goal maps, compute the ratio of observations we want in each filter.
    This is basically:
    per filter, sum the number of pixels in each map and return this per filter value, normalized
    so that the total sum across all filters is 1.
    """
    results = {}
    all_norm = 0.
    for key in target_maps:
        good = target_maps[key] > 0
        results[key] = np.sum(target_maps[key][good])
        all_norm += results[key]
    for key in results:
        results[key] /= all_norm
    return results
