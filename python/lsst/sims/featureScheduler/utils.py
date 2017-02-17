import numpy as np
import healpy as hp
from scipy.spatial import cKDTree as kdtree
from lsst.sims.utils import _hpid2RaDec


def empty_observation():
    """
    Return a numpy array that could be a handy observation record
    """
    names = ['RA', 'Dec', 'mjd', 'exptime', 'filter', 'rotSkyPos']
    # units of rad, rad,   days,  seconds,   string, radians (E of N?)
    types = [float, float, float, float, '|1S', float]
    result = np.zeros(1, dtype=zip(names, types))
    return result


def treexyz(ra, dec):
    """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def hp_kd_tree(nside=32, leafsize=100):
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
    def __init__(self, nside=32, fov_radius=1.75):
        self.tree = hp_kd_tree()
        self.radius = rad_length(fov_radius)

    def __call__(self, ra, dec):
        """
        ra dec in radians
        """
        x, y, z = treexyz(ra, dec)
        indices = self.tree.query_ball_point((x,y,z), self.radius)
        return indices
