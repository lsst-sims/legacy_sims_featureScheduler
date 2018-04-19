from __future__ import print_function
from builtins import zip
from builtins import object
import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.path as mplPath
import logging
from scipy.spatial import cKDTree as kdtree
from lsst.sims.utils import _hpid2RaDec, calcLmstLast, _raDec2Hpid
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import sys
from lsst.utils import getPackageDir
import sqlite3 as db
import matplotlib.pylab as plt
import time
import datetime
from . import version
import warnings

_load_local_fieldlist = False
try:
    from lsst.ts.scheduler.fields import FieldsDatabase
except ImportError as err:
    warnings.warn('''Could not import ts.scheduler. This is required to load the FieldsDatabase. In this case
it will fallback to loading fields from the local "fieldID.lis" file.''')
    _load_local_fieldlist = True

def set_default_nside(nside=None):
    """
    Utility function to set a default nside value across the scheduler.

    XXX-there might be a better way to do this.

    Parameters
    ----------
    nside : int (None)
        A valid healpixel nside.
    """
    if not hasattr(set_default_nside, 'nside'):
        if nside is None:
            nside = 64
        set_default_nside.nside = nside
    if nside is not None:
        set_default_nside.nside = nside
    return set_default_nside.nside


def gnomonic_project_toxy(RA1, Dec1, RAcen, Deccen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccen.
    Input radians. Grabbed from sims_selfcal"""
    # also used in Global Telescope Network website
    cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
    x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
    y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, RAcen, Deccen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(Deccen) - y * np.sin(Deccen)
    RA = RAcen + np.arctan2(x, denom)
    Dec = np.arctan2(np.sin(Deccen) + y * np.cos(Deccen), np.sqrt(x*x + denom*denom))
    return RA, Dec


def raster_sort(x0, order=['x', 'y'], xbin=1.):
    """Do a sort to scan a grid up and down. Simple starting guess to traveling salesman.

    Parameters
    ----------
    x0 : array
    order : list
        Keys for the order x0 should be sorted in.
    xbin : float (1.)
        The binsize to round off the first coordinate into

    returns
    -------
    array sorted so that it rasters up and down.
    """
    coords = x0.copy()
    bins = np.arange(coords[order[0]].min()-xbin/2., coords[order[0]].max()+3.*xbin/2., xbin)
    # digitize my bins
    coords[order[0]] = np.digitize(coords[order[0]], bins)
    order1 = np.argsort(coords, order=order)
    coords = coords[order1]
    places_to_invert = np.where(np.diff(coords[order[-1]]) < 0)[0]
    if np.size(places_to_invert) > 0:
        places_to_invert += 1
        indx = np.arange(coords.size)
        index_sorted = np.zeros(indx.size, dtype=int)
        index_sorted[0:places_to_invert[0]] = indx[0:places_to_invert[0]]

        for i, inv_pt in enumerate(places_to_invert[:-1]):
            if i % 2 == 0:
                index_sorted[inv_pt:places_to_invert[i+1]] = indx[inv_pt:places_to_invert[i+1]][::-1]
            else:
                index_sorted[inv_pt:places_to_invert[i+1]] = indx[inv_pt:places_to_invert[i+1]]

        if np.size(places_to_invert) % 2 != 0:
            index_sorted[places_to_invert[-1]:] = indx[places_to_invert[-1]:][::-1]
        else:
            index_sorted[places_to_invert[-1]:] = indx[places_to_invert[-1]:]
        return order1[index_sorted]
    else:
        return order1


def empty_observation():
    """
    Return a numpy array that could be a handy observation record

    XXX:  Should this really be "empty visit"? Should we have "visits" made
    up of multple "observations" to support multi-exposure time visits?

    XXX-Could add a bool flag for "observed". Then easy to track all proposed
    observations. Could also add an mjd_min, mjd_max for when an observation should be observed.
    That way we could drop things into the queue for DD fields.

    XXX--might be nice to add a generic "sched_note" str field, to record any metadata that
    would be useful to the scheduler once it's observed. and/or observationID.

    Returns
    -------
    numpy array

    Notes
    -----
    The numpy fields have the following structure
    RA : float
       The Right Acension of the observation (center of the field) (Radians)
    dec : float
       Declination of the observation (Radians)
    mjd : float
       Modified Julian Date at the start of the observation (time shutter opens)
    exptime : float
       Total exposure time of the visit (seconds)
    filter : str
        The filter used. Should be one of u, g, r, i, z, y.
    rotSkyPos : float
        The rotation angle of the camera relative to the sky E of N (Radians)
    nexp : int
        Number of exposures in the visit.
    airmass : float
        Airmass at the center of the field
    FWHMeff : float
        The effective seeing FWHM at the center of the field. (arcsec)
    skybrightness : float
        The surface brightness of the sky background at the center of the
        field. (mag/sq arcsec)
    night : int
        The night number of the observation (days)
    """
    names = ['RA', 'dec', 'mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp',
             'airmass', 'FWHMeff', 'FWHM_geometric', 'skybrightness', 'night', 'slewtime', 'fivesigmadepth',
             'alt', 'az', 'clouds', 'moonAlt', 'sunAlt', 'note', 'field_id', 'survey_id']
    # units of rad, rad,   days,  seconds,   string, radians (E of N?)
    types = [float, float, float, float, '|U1', float, int, float, float, float, float, int, float, float,
             float, float, float, float, float, '|U40', int, int]
    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


def empty_scheduled_observation():
    """
    Same as empty observation, but with mjd_min, mjd_max columns
    """
    start = empty_observation()
    names = start.dtype.names
    types = start.dtype.types
    names.extend(['mjd_min', 'mjd_max'])
    types.extend([float, float])

    result = np.zeros(1, dtype=list(zip(names, types)))
    return result


def read_fields():
    """
    Read in the Field coordinates
    Returns
    -------
    numpy.array
        With RA and dec in radians.
    """
    if _load_local_fieldlist:
        return read_fields_from_localfile()
    else:
        return read_fields_from_tscheduler()


def read_fields_from_localfile():
    names = ['RA', 'dec']
    types = [float, float]
    data_dir = os.path.join(getPackageDir('sims_featureScheduler'), 'python/lsst/sims/featureScheduler/')
    filepath = os.path.join(data_dir, 'fieldID.lis')
    field_coords = np.loadtxt(filepath, dtype=list(zip(names, types)))

    field_names = ['field_id', 'fov_rad', 'RA', 'dec', 'gl', 'gb', 'el', 'eb', 'tag']
    field_types = [int, float, float, float, float, float, float, float, int]
    fields = np.zeros(len(field_coords['RA']), dtype=list(zip(field_names, field_types)))

    fields['RA'] = np.radians(field_coords['RA'])
    fields['dec'] = np.radians(field_coords['dec'])

    return fields


def read_fields_from_tscheduler():
    sql = "select * from Field"
    db = FieldsDatabase()
    res = db.query(sql)
    names = ['field_id', 'fov_rad', 'RA', 'dec', 'gl', 'gb', 'el', 'eb', 'tag']
    types = [int, float, float, float, float, float, float, float, int]
    fields = np.zeros(len(res), dtype=list(zip(names, types)))

    for i, row in enumerate(res):
        fields['field_id'][i] = row[0]
        fields['fov_rad'][i] = row[1]
        fields['RA'][i] = np.radians(row[2])
        fields['dec'][i] = np.radians(row[3])
        fields['gl'][i] = row[4]
        fields['gb'][i] = row[5]
        fields['el'][i] = row[6]
        fields['eb'][i] = row[7]

    return fields


def treexyz(ra, dec):
    """
    Utility to convert RA,dec postions in x,y,z space, useful for constructing KD-trees.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def xyz2radec(x, y, z):
    """
    Convert x, y, z coords back to ra and dec in radians (dec=theta, ra=phi)
    """
    r = (x*2 + y**2 + z**2)**0.5
    ra = np.arctan2(y, x)
    dec = np.arccos(z/r)
    return ra, dec


def hp_kd_tree(nside=None, leafsize=100):
    """
    Generate a KD-tree of healpixel locations

    Parameters
    ----------
    nside : int
        A valid healpix nside
    leafsize : int (100)
        Leafsize of the kdtree

    Returns
    -------
    tree : scipy kdtree
    """
    if nside is None:
        nside = set_default_nside()

    hpid = np.arange(hp.nside2npix(nside))
    ra, dec = _hpid2RaDec(nside, hpid)
    x, y, z = treexyz(ra, dec)
    tree = kdtree(list(zip(x, y, z)), leafsize=leafsize, balanced_tree=False, compact_nodes=False)
    return tree


def rad_length(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

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
    Return the healpixels within a pointing. A very simple LSST camera model with
    no chip/raft gaps.
    """
    def __init__(self, nside=None, fov_radius=1.75):
        """
        Parameters
        ----------
        fov_radius : float (1.75)
            Radius of the filed of view in degrees
        """
        if nside is None:
            nside = set_default_nside()

        self.tree = hp_kd_tree(nside=nside)
        self.radius = rad_length(fov_radius)

    def __call__(self, ra, dec, *args):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians

        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = treexyz(np.max(ra), np.max(dec))
        indices = self.tree.query_ball_point((x, y, z), self.radius)
        return np.array(indices)


class hp_in_comcam_fov(object):
    """
    Return the healpixels within a ComCam pointing. Simple camera model
    with no chip gaps.
    """
    def __init__(self, nside=None, side_length=0.7):
        """
        Parameters
        ----------
        side_length : float (0.7)
            The length of one side of the square field of view (degrees).
        """
        if nside is None:
            nside = set_default_nside()
        self.nside = nside
        self.tree = hp_kd_tree(nside=nside)
        self.side_length = np.radians(side_length)
        self.inner_radius = rad_length(side_length/2.)
        self.outter_radius = rad_length(side_length/2.*np.sqrt(2.))
        # The positions of the raft corners, unrotated
        self.corners_x = np.array([-self.side_length/2., -self.side_length/2., self.side_length/2.,
                                  self.side_length/2.])
        self.corners_y = np.array([self.side_length/2., -self.side_length/2., -self.side_length/2.,
                                  self.side_length/2.])

    def __call__(self, ra, dec, rotSkyPos):
        """
        Parameters
        ----------
        ra : float
            RA in radians
        dec : float
            Dec in radians
        rotSkyPos : float
            The rotation angle of the camera in radians
        Returns
        -------
        indx : numpy array
            The healpixels that are within the FoV
        """
        x, y, z = treexyz(np.max(ra), np.max(dec))
        # Healpixels within the inner circle
        indices = self.tree.query_ball_point((x, y, z), self.inner_radius)
        # Healpixels withing the outer circle
        indices_all = np.array(self.tree.query_ball_point((x, y, z), self.outter_radius))
        indices_to_check = indices_all[np.in1d(indices_all, indices, invert=True)]

        cos_rot = np.cos(rotSkyPos)
        sin_rot = np.sin(rotSkyPos)
        x_rotated = self.corners_x*cos_rot - self.corners_y*sin_rot
        y_rotated = self.corners_x*sin_rot + self.corners_y*cos_rot

        # Draw the square that we want to check if points are in.
        bbPath = mplPath.Path(np.array([[x_rotated[0], y_rotated[0]],
                                       [x_rotated[1], y_rotated[1]],
                                       [x_rotated[2], y_rotated[2]],
                                       [x_rotated[3], y_rotated[3]],
                                       [x_rotated[0], y_rotated[0]]]))

        ra_to_check, dec_to_check = _hpid2RaDec(self.nside, indices_to_check)

        # Project the indices to check to the tangent plane, see if they fall inside the polygon
        x, y = gnomonic_project_toxy(ra_to_check, dec_to_check, ra, dec)
        for i, xcheck in enumerate(x):
            # I wonder if I can do this all at once rather than a loop?
            if bbPath.contains_point((x[i], y[i])):
                indices.append(indices_to_check[i])

        return np.array(indices)


def ra_dec_hp_map(nside=None):
    """
    Return all the RA,dec points for the centers of a healpix map
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec


def WFD_healpixels(nside=None, dec_min=-60., dec_max=0.):
    """
    Define a wide fast deep region. Return a healpix map with WFD pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    result[good] += 1
    return result

def WFD_upper_edge_healpixels(nside=None, dec_min=2.8, dec_max=None):
    """
    Define a strip at the northern edge of the WFD area.

    Parameters
    ----------
    nside : int
        A valid healpix nside
    dec_min : float (2.8)
        Minimum dec of the strip (deg)
    dec_max : float (None)
        Maximum dec of strip (deg). If left None dec_max is dec_min + 3.*1.75.
        1.75 is the FOV radius in deg.
    Returns
    -------
    result : numpy array
    """
    if nside is None:
        nside = set_default_nside()

    if dec_max is None:
        dec_max = dec_min + 3.*1.75

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    result[good] += 1
    return result


def SCP_healpixels(nside=None, dec_max=-60.):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()

    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where(dec < np.radians(dec_max))
    result[good] += 1
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
    result = np.zeros(ra.size)
    coord = SkyCoord(ra=ra*u.rad, dec=dec*u.rad)
    eclip_lat = coord.barycentrictrueecliptic.lat.radian
    good = np.where((eclip_lat > np.radians(min_EB)) &
                    (eclip_lat < np.radians(max_EB)) &
                    (dec > np.radians(dec_min)))
    result[good] += 1

    return result


def galactic_plane_healpixels(nside=None, center_width=10., end_width=4.,
                              gal_long1=70., gal_long2=290.):
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as 1.
    """
    if nside is None:
        nside = set_default_nside()

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


def generate_goal_map(nside=None, NES_fraction = .3, WFD_fraction = 1., SCP_fraction=0.4,
                      GP_fraction = 0.2, WFD_upper_edge_fraction = 0.25,
                      NES_min_EB = -30., NES_max_EB = 10, NES_dec_min = 2.8,
                      SCP_dec_max=-62.5, gp_center_width=10.,
                      gp_end_width=4., gp_long1=70., gp_long2=290.,
                      wfd_dec_min=-62.5, wfd_dec_max=2.8,
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

    if WFD_upper_edge_fraction > 0.:
        wfd_upper_edge = WFD_upper_edge_healpixels(nside=nside, dec_min=wfd_dec_max)
        result[np.where(wfd_upper_edge != 0)] = 0
        result += WFD_upper_edge_fraction*wfd_upper_edge
        id_map[np.where(wfd_upper_edge != 0)] = 3
        pid += 1
        prop_name_dict[3] = 'WideFastDeep'


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

def generate_cloud_map(target_maps=None, filtername='r', wfd_cloud_max=0.7,
                       scp_cloud_max=0.7, gp_cloud_max=0.7, nes_cloud_max=0.7):
    """
    Generate maximum cloud map from a target map produced by generate_goal_map.

    Parameters
    ----------
    target_maps : dict (None)
        A dictionary containing healpix goal maps returned by generate_goal_map.
    filtername : str
        Name of filter.
    wfd_cloud_max : float
        Maximum cloud value for wide fast deep
    scp_cloud_max : float
        Maximum cloud values for south celestial pole
    gp_cloud_max : float
        Maximum cloud values for galactic plane
    nes_cloud_max : float
        Maximum cloud values for north ecliptic spur

    Returns
    -------
    cloud_map : numpy array
        Healpix where each pixel is assigned a maximum allowed value
    """
    cloud_map = np.zeros_like(target_maps[filtername][0])

    wfd_cloud = np.where(target_maps[filtername][1] == 3)
    scp_cloud = np.where(target_maps[filtername][1] == 2)
    gp_cloud = np.where(target_maps[filtername][1] == 4)
    nes_cloud = np.where(target_maps[filtername][1] == 1)

    cloud_map[wfd_cloud] = wfd_cloud_max
    cloud_map[scp_cloud] = scp_cloud_max
    cloud_map[gp_cloud]  = gp_cloud_max
    cloud_map[nes_cloud] = nes_cloud_max

    return cloud_map

def standard_goals(nside=None):
    """
    A quick function to generate the "standard" goal maps.
    """
    # Find the number of healpixels we expect to observe per observation
    if nside is None:
        nside = set_default_nside()

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


def filter_count_ratios(target_maps):
    """Given the goal maps, compute the ratio of observations we want in each filter.
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


def run_info_table(observatory):
    """
    Make a little table for recording the information of a run
    """
    names = ['time', 'datetime', 'ymd', 'version', 'fingerprint', 'observatory_class', 'obs_finger']
    types = [float, '|U20', '|U20', '|U20', '|U50', '|U20', '|U50']
    result = np.zeros(1, dtype=list(zip(names, types)))
    result['time'] = np.float(time.time())
    now = datetime.datetime.now()
    result['ymd'] = '%i, %i, %i' % (now.year, now.month, now.day)
    result['version'] = version.__version__
    result['fingerprint'] = version.__fingerprint__
    result['observatory_class'] = observatory.__class__.__name__

    try:
        result['obs_finger'] = observatory.version.__fingerprint__
    except:
        pass
    return result


def sim_runner(observatory, scheduler, mjd_start=None, survey_length=3., filename=None, delete_past=True):
    """
    run a simulation
    """

    if mjd_start is None:
        mjd = observatory.mjd
        mjd_start = mjd + 0
    else:
        observatory.mjd = mjd
        observatory.ra = None
        observatory.dec = None
        observatory.status = None
        observatory.filtername = None

    end_mjd = mjd + survey_length
    scheduler.update_conditions(observatory.return_status())
    observations = []
    mjd_track = mjd + 0
    step = 1./24.
    step_none = 1./60./24.  # 1 minute in days
    mjd_run = end_mjd-mjd_start
    nskip = 0

    while mjd < end_mjd:
        desired_obs = scheduler.request_observation()
        if desired_obs is None:
            # No observation. Just step into the future and try again.
            warnings.warn('No observation. Step into the future and trying again.')
            observatory.mjd += step_none
            scheduler.update_conditions(observatory.return_status())
            nskip += 1
            continue

        attempted_obs = observatory.attempt_observe(desired_obs)
        if attempted_obs is not None:
            scheduler.add_observation(attempted_obs[0])
            observations.append(attempted_obs)
        else:
            scheduler.flush_queue()
        scheduler.update_conditions(observatory.return_status())
        mjd = observatory.mjd
        if (mjd-mjd_track) > step:
            progress = float(mjd-mjd_start)/mjd_run*100
            text = "\rprogress = %.1f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()
            mjd_track = mjd+0
        # XXX--handy place to interupt and debug
        #if len(observations) > 3:
        #    import pdb ; pdb.set_trace()

    print('Skipped %i observations' % nskip)
    print('Completed %i observations' % len(observations))
    observations = np.array(observations)[:, 0]
    if filename is not None:
        # don't crash just because some info stuff failed.
        try:
            info = run_info_table(observatory)
        except:
            info = None
            warnings.warn('Failed to get info about run, may need to run scons in some pacakges.')
        observations2sqlite(observations, filename=filename, delete_past=delete_past, info=info)
    return observatory, scheduler, observations


def observations2sqlite(observations, filename='observations.db', delete_past=False, info=None):
    """
    Take an array of observations and dump it to a sqlite3 database

    Parameters
    ----------
    observations : numpy.array
        An array of executed observations
    filename : str (observations.db)
        Filename to save sqlite3 to. Value of None will skip
        writing out file.
    delete_past : bool (False)
        If True, overwrite any previous file with the same fileaname.
    info : np.array (None)
        A numpy array of information about the run.

    Returns
    -------
    observations : numpy.array
        The observations array updated to have angles in degrees and
        any added columns
    """

    # XXX--Here is a good place to add any missing columns, e.g., alt,az

    if delete_past:
        try:
            os.remove(filename)
        except OSError:
            pass

    # Convert to degrees for output
    to_convert = ['RA', 'dec', 'alt', 'az', 'rotSkyPos', 'moonAlt', 'sunAlt']
    for key in to_convert:
        observations[key] = np.degrees(observations[key])

    if filename is not None:
        df = pd.DataFrame(observations)
        con = db.connect(filename)
        df.to_sql('observations', con, index_label='observationId')
        if info is not None:
            df = pd.DataFrame(info)
            df.to_sql('info', con)
    return observations


def sqlite2observations(filename='observations.db'):
    """
    Restore a databse of observations.
    """
    con = db.connect(filename)
    df = pd.read_sql('select * from observations;', con)
    blank = empty_observation()
    result = df.as_matrix()
    final_result = np.empty(result.shape[0], dtype=blank.dtype)

    # XXX-ugh, there has to be a better way.
    for i, key in enumerate(blank.dtype.names):
        final_result[key] = result[:, i+1]

    to_convert = ['RA', 'dec', 'alt', 'az', 'rotSkyPos', 'moonAlt', 'sunAlt']
    for key in to_convert:
        final_result[key] = np.radians(final_result[key])

    return final_result


def inrange(inval, minimum=-1., maximum=1.):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval


def warm_start(scheduler, observations, mjd_key='mjd'):
    """Replay a list of observations into the scheduler

    Parameters
    ----------
    scheduler : scheduler object

    observations : np.array
        An array of observation (e.g., from sqlite2observations)
    """

    # Check that observations are in order
    observations.sort(order=mjd_key)
    for observation in observations:
        scheduler.add_observation(observation)

    return scheduler


def stupidFast_altAz2RaDec(alt, az, lat, lon, mjd):
    """
    Convert alt, az to RA, Dec without taking into account abberation, precesion, diffraction, ect.

    Parameters
    ----------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Must be same length as `alt`. Radians.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians.
    """
    lmst, last = calcLmstLast(mjd, lon)
    lmst = lmst/12.*np.pi  # convert to rad
    sindec = np.sin(lat)*np.sin(alt) + np.cos(lat)*np.cos(alt)*np.cos(az)
    sindec = inrange(sindec)
    dec = np.arcsin(sindec)
    ha = np.arctan2(-np.sin(az)*np.cos(alt), -np.cos(az)*np.sin(lat)*np.cos(alt)+np.sin(alt)*np.cos(lat))
    ra = (lmst-ha)
    raneg = np.where(ra < 0)
    ra[raneg] = ra[raneg] + 2.*np.pi
    raover = np.where(ra > 2.*np.pi)
    ra[raover] -= 2.*np.pi
    return ra, dec


def stupidFast_RaDec2AltAz(ra, dec, lat, lon, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore abberation, precesion, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    if lmst is None:
        lmst, last = calcLmstLast(mjd, lon)
        lmst = lmst/12.*np.pi  # convert to rad
    ha = lmst-ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec*sinlat+np.cos(dec)*coslat*np.cos(ha)
    sinalt = inrange(sinalt)
    alt = np.arcsin(sinalt)
    cosaz = (sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat)
    cosaz = inrange(cosaz)
    az = np.arccos(cosaz)
    signflip = np.where(np.sin(ha) > 0)
    az[signflip] = 2.*np.pi-az[signflip]
    return alt, az
