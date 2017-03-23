import numpy as np
import healpy as hp
from .utils import treexyz, hp_kd_tree, rad_length


def wrapRADec(ra, dec):
    # XXX--from MAF, should put in general utils
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def ra_dec_2_xyz(ra, dec):
        """Calculate x/y/z values for ra/dec points, ra/dec in radians."""
        # Note ra/dec can be arrays.
        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        return x, y, z


def rotate_ra_dec(ra, dec, ra_target, dec_target, init_rotate=0.):
    """
    Rotate ra and dec coordinates to be centered on a new dec.

    Inputs
    ------
    ra : float or np.array
        RA coordinate(s) to be rotated in radians
    dec : float or np.array
        Dec coordinate(s) to be rotated in radians
    ra_rotation : float
        RA distance to rotate in radians
    dec_target : float
        Dec distance to rotate in radians
    init_rotate : float (0.)
        The amount to rotate the points around the x-axis first (radians).
    """
    # point (ra,dec) = (0,0) is at x,y,z = 1,0,0

    x, y, z = treexyz(ra, dec)

    # Rotate around the x axis to start
    xp = x
    if init_rotate != 0.:
        c_i = np.cos(init_rotate)
        s_i = np.sin(init_rotate)
        yp = c_i*y - s_i*z
        zp = s_i*y + c_i*z
    else:
        yp = y
        zp = z

    theta_y = dec_target
    c_ty = np.cos(theta_y)
    s_ty = np.sin(theta_y)

    # Rotate about y
    xp2 = c_ty*xp + s_ty*zp
    zp2 = -s_ty*xp + c_ty*zp

    # Convert back to RA, Dec
    ra_p = np.arctan2(yp, xp2)
    dec_p = np.arcsin(zp2)

    # Rotate to the correct RA
    ra_p += ra_target

    ra_p, dec_p = wrapRADec(ra_p, dec_p)

    return ra_p, dec_p


class pointings2hp(object):
    """
    Convert a list of telescope pointings and convert them to a pointing map
    """
    def __init__(self, nside, radius=1.75):
        """

        """
        self.tree = hp_kd_tree(nside=nside)
        self.nside = nside
        self.rad = rad_length(radius)
        self.bins = np.arange(hp.nside2npix(nside)+1)-.5

    def __call__(self, ra, dec):
        """
        similar to utils.hp_in_lsst_fov, but can take a arrays of ra,dec.

        Parameters
        ----------
        ra : array_like
            RA in radians
        dec : array_like
            Dec in radians

        Returns
        -------
        result : healpy map
            The number of times each healpxel is observed by the given pointings
        """
        xs, ys, zs = treexyz(ra, dec)
        coords = np.array(xs, ys, zs).T
        indx = self.tree.query_ball_point(coords, self.rad)
        # Convert array of lists to single array
        indx = np.hstack(indx)
        result, bins = np.histogram(indx, bins=self.bins)
        return result







