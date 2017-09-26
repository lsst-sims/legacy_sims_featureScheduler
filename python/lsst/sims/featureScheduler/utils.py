from __future__ import print_function
from builtins import zip
from builtins import object
import numpy as np
import healpy as hp
import pandas as pd
from scipy.spatial import cKDTree as kdtree
from lsst.sims.utils import _hpid2RaDec, calcLmstLast, _raDec2Hpid
from astropy.coordinates import SkyCoord
from astropy import units as u
import os
import sys
from lsst.utils import getPackageDir
import sqlite3 as db
from lsst.sims.utils import haversine


def set_default_nside(nside=None):
    """
    Utility function to set a default nside value across the scheduler.

    Parameters
    ----------
    nside : int (None)
        A valid healpixel nside.
    """
    if not hasattr(set_default_nside, 'nside'):
        if nside is None:
            nside = 64
        set_default_nside.side = nside
    return set_default_nside.side


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
        return x0[order1][index_sorted]
    else:
        return x0[order1]


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
             'alt', 'az', 'clouds']
    # units of rad, rad,   days,  seconds,   string, radians (E of N?)
    types = [float, float, float, float, '|U1', float, int, float, float, float, float, int, float, float,
             float, float, float]
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
    Read in the old Field coordinates
    Returns
    -------
    numpy.array
        With RA and dec in radians.
    """
    names = ['id', 'RA', 'dec']
    types = [int, float, float]
    data_dir = os.path.join(getPackageDir('sims_featureScheduler'), 'python/lsst/sims/featureScheduler/')
    filepath = os.path.join(data_dir, 'fieldID.lis')
    fields = np.loadtxt(filepath, dtype=list(zip(names, types)))
    fields['RA'] = np.radians(fields['RA'])
    fields['dec'] = np.radians(fields['dec'])
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


def hp_kd_tree(nside=set_default_nside(), leafsize=100):
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
    def __init__(self, nside=set_default_nside(), fov_radius=1.75):
        """
        Parameters
        ----------
        fov_radius : float (1.75)
            Radius of the filed of view in degrees
        """
        self.tree = hp_kd_tree(nside=nside)
        self.radius = rad_length(fov_radius)

    def __call__(self, ra, dec):
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


def ra_dec_hp_map(nside=set_default_nside()):
    """
    Return all the RA,dec points for the centers of a healpix map
    """
    ra, dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
    return ra, dec


def WFD_healpixels(nside=set_default_nside(), dec_min=-60., dec_max=0.):
    """
    Define a wide fast deep region. Return a healpix map with WFD pixels as 1.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    result[good] += 1
    return result


def SCP_healpixels(nside=set_default_nside(), dec_max=-60.):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as 1.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where(dec < np.radians(dec_max))
    result[good] += 1
    return result


def NES_healpixels(nside=set_default_nside(), width=15, dec_min=0., fill_gap=True):
    """
    Define the North Ecliptic Spur region. Return a healpix map with NES pixels as 1.
    """
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
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as 1.
    """
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

    # Note, some regions overlap, thus order regions are added is important.
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


def sim_runner(observatory, scheduler, mjd_start=None, survey_length=3., filename=None, delete_past=False):
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
    mjd_run = end_mjd-mjd_start

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
        if (mjd-mjd_track) > step:
            progress = float(mjd-mjd_start)/mjd_run*100
            text = "\rprogress = %.1f%%" % progress
            sys.stdout.write(text)
            sys.stdout.flush()
            mjd_track = mjd+0

    print('Completed %i observations' % len(observations))
    observations = np.array(observations)[:, 0]
    if filename is not None:
        observations2sqlite(observations, filename=filename, delete_past=delete_past)
    return observatory, scheduler, observations


def observations2sqlite(observations, filename='observations.db', delete_past=False):
    """
    Take an array of observations and dump it to a sqlite3 database

    Parameters
    ----------
    observations : numpy.array
        An array of executed observations
    filename : str (observations.db)
        Filename to save sqlite3 to. Value of None will skip
        writing out file.

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
    observations['RA'] = np.degrees(observations['RA'])
    observations['dec'] = np.degrees(observations['dec'])
    observations['alt'] = np.degrees(observations['alt'])
    observations['az'] = np.degrees(observations['az'])
    observations['rotSkyPos'] = np.degrees(observations['rotSkyPos'])

    if filename is not None:
        df = pd.DataFrame(observations)
        con = db.connect(filename)
        df.to_sql('SummaryAllProps', con, index_label='observationId')
    return observations


def sqlite2observations(filename='observations.db'):
    """
    Restore a databse of observations.
    """
    con = db.connect(filename)
    df = pd.read_sql('select * from SummaryAllProps;', con)
    return df


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


def sort_pointings(observations, order_first='azimuth'):
    """
    Try to sort a group of pointings to be executed in a good order

    Parameters
    ----------
    observations : list of observation objects
        The observations we want to sort.
    order : str (azimuth)
        Sort by azimuth or altitude first.

    Returns
    -------
    The observations sorted in a good order
    """

    obs_array = np.array(observations)[:, 0]
    # compute alt-az and raster in the correct way.
    # Maybe take some windows. 
    # Note that the az rotation is a problem near zenith. 
    # does a greedy walk do a good job?
    return


def max_altitude(dec, lsst_lat):
    """
    evaluates the maximum altitude that fields can ever achieve.

    Parameters
    ----------
    dec : numpy.array
        declination of the fields
    lsst_lat  : float
        Lattitude of the LSST site

    Returns
    -------
    max_alt : numpy.array
        Maximum altitudes. Radians.
    """
    max_alt = lsst_lat + np.pi/2. - dec
    max_alt = np.where(dec >= lsst_lat, lsst_lat + np.pi/2. - dec, -lsst_lat + np.pi/2. + dec)
    return max_alt


def alt_allocation(alt, dec, lsst_lat, filter_name='r'):
    """
    Allocates altitude to each filter, so there is a best normalized altitude for each filter

    Parameters
    ----------
    alt : numpy.array
        altitude of the fields
    dec : numpy.array
        declination of the fields (to find the maximum altitude they can ever reach)
    lsst_lat  : float
        Lattitude of the LSST site

    Returns
    -------
    alt_alloc : numpy.array

    """
    max_alt = max_altitude(dec, lsst_lat)
    normalized_alt = alt/max_alt
    if filter_name is None:
        filter_name = 'r'
    index = ['u', 'g', 'r', 'i', 'z', 'y'].index(filter_name)
    traps = np.array([0.95,0.85,0.75,0.65,0.55,0.45])

    alt_alloc = 10. * np.square(normalized_alt - traps[index])
    alt_alloc[normalized_alt >= .95] = .95
    alt_alloc[normalized_alt <= .45] = .45
    return alt_alloc


def hour_angle(ra, lsst_lon, mjd, lmst=None):
    """
    evaluates the hour angle of fields.

    Parameters
    ----------
    ra : numpy.array
        RA, in radians.
    lsst_lon  : float
        Longitude of the LSST site

    Returns
    -------
    ha : numpy.array
        Hour angle ranging from -12 to 12. Hours.
    """
    if lmst is None:
        lmst, last = calcLmstLast(mjd, lsst_lon)
    ha = lmst-ra * 12./np.pi
    ha = np.where(ha < -12, ha + 24, ha)
    ha = np.where(ha > 12, ha - 24, ha)
    return ha


def mutually_exclusive_regions(nside=set_default_nside()):
    indx = np.arange(hp.nside2npix(nside))
    all_true = np.ones(np.size(indx), dtype=bool)
    SCP_indx = is_SCP(nside)
    NES_indx = is_NES(nside)
    GP_indx = is_GP(nside)

    all_butWFD = SCP_indx + NES_indx + GP_indx
    GP_NES = GP_indx + NES_indx

    WFD_indx = all_true - all_butWFD
    SCP_indx = SCP_indx - GP_NES*SCP_indx - NES_indx*SCP_indx
    NES_indx = NES_indx - GP_indx*SCP_indx

    return indx[SCP_indx], indx[NES_indx], indx[GP_indx], indx[WFD_indx]


def is_WFD(nside=set_default_nside(), dec_min=-60., dec_max=0.):
    """
    Define a wide fast deep region. Return a healpix map with WFD pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    WFD_indx = ((dec >= np.radians(dec_min)) & (dec <= np.radians(dec_max)))
    return WFD_indx


def is_SCP(nside=set_default_nside(), dec_max=-60.):
    """
    Define the South Celestial Pole region. Return a healpix map with SCP pixels as true.
    """
    ra, dec = ra_dec_hp_map(nside=nside)
    result = np.zeros(ra.size)
    good = np.where(dec < np.radians(dec_max))
    result[good] += 1

    SCP_indx = (result == 1)
    return SCP_indx


def is_NES(nside=set_default_nside(), width=15, dec_min=0., fill_gap=True):
    """
    Define the North Ecliptic Spur region. Return a healpix map with NES pixels as true.
    """
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

    NES_indx = (result == 1)

    return NES_indx


def is_GP(nside=set_default_nside(), center_width=10., end_width=4.,
          gal_long1=70., gal_long2=290.):
    """
    Define the Galactic Plane region. Return a healpix map with GP pixels as true.
    """
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

    GP_inx = (result == 1)

    return GP_inx


def RaDec2region(ra, dec, nside):

    result = np.array(np.size(ra), dtype = str)
    SCP_indx, NES_indx, GP_indx, WFD_indx = mutually_exclusive_regions(nside)
    indices = _raDec2Hpid(ra, dec, nside)

    SCP = np.searchsorted(indices, SCP_indx)
    NES = np.searchsorted(indices, NES_indx)
    GP = np.searchsorted(indices, GP_indx)
    WFD = np.searchsorted(indices, WFD_indx)

    result[SCP] = 'SCP'
    result[NES] = 'NES'
    result[GP] = 'GP'
    result[WFD] = 'WFD'

    return result


def simple_performance_measure(observations, preferences):
    # survey length
    t_decimals = np.modf(observations['mjd'])[0]
    t_dec_shifted = np.roll(t_decimals, -1)
    deltas = t_dec_shifted - t_decimals
    interval = np.sum(deltas[deltas > 0]) * 24.*60.*60.
    # avg of slew times
    avg_slew = np.sum(observations['slewtime'])/interval
    # performance
    return preferences[0] * (1-avg_slew)


def slew_time(ra0, dec0, ra1, dec1):
    ang_speed = np.radians(5.)
    """
    Compute slew time to new ra, dec position
    """
    dist = haversine(ra1, dec1, ra0, dec0)
    time = dist / ang_speed
    return time


