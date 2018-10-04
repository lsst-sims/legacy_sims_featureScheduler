import numpy as np
from lsst.sims.utils import _approx_RaDec2AltAz, Site, _hpid2RaDec
import healpy as hp


__all__ = ['Conditions']


class Conditions(object):
    """
    Class to hold telemetry information
    """
    def __init__(self, nside, site='LSST'):
        """
        Parameters
        ----------

        """

        self.nside = nside
        self.site = Site(site)
        hpids = np.arange(hp.nside2npix(nside))
        # The RA, Dec grid we are using
        self.ra_rad, self.dec_rad = _hpid2RaDec(nside, hpids)

        # Modified Julian Date (day)
        self.mjd = None
        # Altitude and azimuth. Dict with degrees and radians
        self.altAz = None
        # The cloud level. Fraction, but could upgrade to transparency map
        self.clouds = None
        

    @property
    def altAz(self):
        if self._azAlt is None:
            self.calc_altAz()
        return self._altAz

    
    def calc_altAz(self):
        alt, az = _approx_RaDec2AltAz(self.ra_rad, self.dec_rad,
                                      self.site.latitude_rad,
                                      self.longitude_rad, self._mjd)
        self._altAz = {'alt_rad': alt, 'az_rad': az,
                       'alt': np.degrees(alt), 'az': np.degrees(az)}

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        # Set things that need to be recalculated to None
        self._azAlt = None
