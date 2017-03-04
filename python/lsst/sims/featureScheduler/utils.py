import numpy as np
import healpy as hp
import pandas as pd
from scipy.spatial import cKDTree as kdtree
from lsst.sims.utils import _hpid2RaDec
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
from lsst.utils import getPackageDir
import sqlite3 as db


def set_default_nside(nside=None):
    """
    Can just call this function at the start to set to a different nside for everything
    """
    if not hasattr(set_default_nside, 'nside'):
        if nside is None:
            nside = 64
        set_default_nside.side = nside
    return set_default_nside.side


def empty_observation():
    """
    Return a numpy array that could be a handy observation record

    XXX:  Document all the fields and units!!!
    """
    names = ['RA', 'dec', 'mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
             'airmass', 'FWHMeff', 'skybrightness']
    # units of rad, rad,   days,  seconds,   string, radians (E of N?)
    types = [float, float, float, float, '|1S', float, int, float, float, float]
    result = np.zeros(1, dtype=zip(names, types))
    return result


def read_fields():
    """
    Read in the old Field coordinates
    """
    names = ['id', 'RA', 'dec']
    types = [int, float, float]
    data_dir = os.path.join(getPackageDir('sims_featureScheduler'), 'python/lsst/sims/featureScheduler/')
    filepath = os.path.join(data_dir, 'fieldID.lis')
    fields = np.loadtxt(filepath, dtype=zip(names, types))
    fields['RA'] = np.radians(fields['RA'])
    fields['dec'] = np.radians(fields['dec'])
    return fields


def treexyz(ra, dec):
    """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def hp_kd_tree(nside=set_default_nside(), leafsize=100):
    hpid = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2RaDec(nside, hpid)
    x, y, z = treexyz(ra, dec)
    tree = kdtree(zip(x, y, z), leafsize=leafsize, balanced_tree=False, compact_nodes=False)
    return tree


def rad_length(radius=1.75):
    """
    Parameters
    ----------
    radius : float
        Radius in degrees.
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = treexyz(np.radians(radius), 0)
    result = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return result


class hp_in_lsst_fov(object):
    """
    Return the healpixels within a pointing
    """
    def __init__(self, nside=set_default_nside(), fov_radius=1.75):
        self.tree = hp_kd_tree()
        self.radius = rad_length(fov_radius)

    def __call__(self, ra, dec):
        """
        ra dec in radians. Only single points for now.
        """
        x, y, z = treexyz(np.max(ra), np.max(dec))
        indices = self.tree.query_ball_point((x, y, z), self.radius)
        return np.array(indices)


def ra_dec_hp_map(nside=set_default_nside()):
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec


def WFD_healpixels(nside=set_default_nside(), dec_min=-60., dec_max=0.):
    """
    Define a wide fast deep region.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    result[good] += 1
    return result


def SCP_healpixels(nside=set_default_nside(), dec_max=-60.):
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where(dec < np.radians(dec_max))
    result[good] += 1
    return result


def NES_healpixels(nside=set_default_nside(), width=15, dec_min=0., fill_gap=True):
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where((np.abs(eclip_lat) <= np.radians(width)) & (dec > dec_min))
    result[good] += 1

    if fill_gap:
        good = np.where((dec > np.radians(dec_min)) & (ra < np.radians(180)) &
                        (dec < np.radians(width)))
        result[good] = 1

    return result


def galactic_plane_healpixels(nside=set_default_nside(), center_width=10., end_width=4.,
                              gal_long1=70., gal_long2=290.):
    # XXX--this is not right yet
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    g_long, g_lat = coord.galactic.l.radian, coord.galactic.b.radian
    good = np.where((g_long < np.radians(gal_long1)) & (np.abs(g_lat) < np.radians(center_width)))
    result[good] += 1
    good = np.where((g_long > np.radians(gal_long2)) & (np.abs(g_lat) < np.radians(center_width)))
    result[good] += 1
    # Add tapers
    slope = -(np.radians(center_width)-np.radians(end_width))/(np.radians(gal_long1))
    lat_limit = slope*g_long+np.radians(center_width)
    outside = np.where((g_long < np.radians(gal_long1)) & (np.abs(g_lat) > np.abs(lat_limit)))
    result[outside] = 0
    slope = (np.radians(center_width)-np.radians(end_width))/(np.radians(360. - gal_long2))
    b = np.radians(center_width)-np.radians(360.)*slope
    lat_limit = slope*g_long+b
    outside = np.where((g_long > np.radians(gal_long2)) & (np.abs(g_lat) > np.abs(lat_limit)))
    result[outside] = 0

    return result


def generate_goal_map(nside=set_default_nside(), NES_fraction = .3, WFD_fraction = 1., SCP_fraction=0.4,
                      GP_fraction = 0.2,
                      NES_width=15., NES_dec_min=0., NES_fill=True,
                      SCP_dec_max=-60., gp_center_width=10.,
                      gp_end_width=4., gp_long1=70., gp_long2=290.,
                      wfd_dec_min=-60., wfd_dec_max=0.):
    """
    Handy function that will put together a target map in the proper order.
    """
    result = np.zeros(hp.nside2npix(nside), dtype=float)
    result += NES_fraction*NES_healpixels(nside=nside, width=NES_width,
                                          dec_min=NES_dec_min, fill_gap=NES_fill)
    wfd = WFD_healpixels(nside=nside, dec_min=wfd_dec_min, dec_max=wfd_dec_max)
    result[np.where(wfd != 0)] = 0
    result += WFD_fraction*wfd
    scp = SCP_healpixels(nside=nside, dec_max=SCP_dec_max)
    result[np.where(scp != 0)] = 0
    result += SCP_fraction*scp
    gp = galactic_plane_healpixels(nside=nside, center_width=gp_center_width,
                                   end_width=gp_end_width, gal_long1=gp_long1,
                                   gal_long2=gp_long2)
    result[np.where(gp != 0)] = 0
    result += GP_fraction*gp
    return result


def standard_goals(nside=set_default_nside()):
    """
    A quick fucntion to generate the "standard" goal maps.
    """
    result = {}
    result['u'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.31, SCP_fraction=0.15,
                                    GP_fraction=0.15)
    result['g'] = generate_goal_map(nside=nside, NES_fraction=0.2,
                                    WFD_fraction=0.44, SCP_fraction=0.15,
                                    GP_fraction=0.15)
    result['r'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15)
    result['i'] = generate_goal_map(nside=nside, NES_fraction=0.46,
                                    WFD_fraction=1.0, SCP_fraction=0.15,
                                    GP_fraction=0.15)
    result['z'] = generate_goal_map(nside=nside, NES_fraction=0.4,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15)
    result['y'] = generate_goal_map(nside=nside, NES_fraction=0.,
                                    WFD_fraction=0.9, SCP_fraction=0.15,
                                    GP_fraction=0.15)

    return result


def sim_runner(observatory, scheduler, mjd_start=None, survey_length=3., filename=None):
    """
    run a simulation
    """

    if mjd_start is None:
        mjd = observatory.mjd
    else:
        observatory.mjd = mjd
        observatory.ra = None
        observatory.dec = None
        observatory.status = None
        observatory.filtername = None

    end_mjd = mjd + survey_length
    scheduler.update_conditions(observatory.return_status())
    observations = []
    while mjd < end_mjd:
        desired_obs = scheduler.request_observation()
        attempted_obs = observatory.attempt_observe(desired_obs)
        if attempted_obs is not None:
            scheduler.add_observation(attempted_obs)
            observations.append(attempted_obs)
        else:
            scheduler.flush_queue()
        scheduler.update_conditions(observatory.return_status())
        mjd = observatory.mjd

    print 'Completed %i observations' % len(observations)
    observations = np.array(observations)[:, 0]
    if filename is not None:
        observations2sqlite(observations, filename=filename)
    return observatory, scheduler, observations


def observations2sqlite(observations, filename='observations.db'):
    """
    Take an array of observations and dump it to a sqlite3 database
    """

    # XXX--Here is a good place to add any missing columns

    # Convert to degrees for output
    observations['RA'] = np.degrees(observations['RA'])
    observations['dec'] = np.degrees(observations['dec'])
    observations['rotSkyPos'] = np.degrees(observations['rotSkyPos'])

    df = pd.DataFrame(observations)
    con = db.connect(filename)
    df.to_sql('SummaryAllProps', con, index_label='observationId')

