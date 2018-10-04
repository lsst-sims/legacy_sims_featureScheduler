import numpy as np
from lsst.sims.utils import _approx_RaDec2AltAz, Site, _hpid2RaDec, m5_flat_sed
import healpy as hp
import numpy.ma as ma


__all__ = ['Conditions']


class Conditions(object):
    """
    Class to hold telemetry information
    """
    def __init__(self, nside, site='LSST', expTime=30.):
        """
        Parameters
        ----------
        expTime : float (30)
            The exposure time to assume when computing the 5-sigma limiting depth
        """

        self.nside = nside
        self.site = Site(site)
        self.expTime = expTime
        hpids = np.arange(hp.nside2npix(nside))
        self.zeros_map = np.zeros(hp.nside2npix(nside), dtype=float)
        # The RA, Dec grid we are using
        self.ra_rad, self.dec_rad = _hpid2RaDec(nside, hpids)

        # Modified Julian Date (day)
        self._mjd = None
        # Altitude and azimuth. Dict with degrees and radians
        self._altAz = None
        # The cloud level. Fraction, but could upgrade to transparency map
        self.clouds = None
        self._slewtime = None
        self.current_filter = None
        self.mounted_filters = None
        self.night = None
        self.lmst = None
        # Should be a dict with filtername keys
        self._skybrightness = {}
        self._FWHMeff = {}
        self._M5Depth = None
        self._airmass

    @property
    def slewtime(self):
        return self._slewtime

    @slewtime.setter
    def slewtime(self, value):
        self._slewtime = hp.ud_grade(value, nside_out=self.nside)

    @property
    def airmass(self):
        return self._airmass

    @airmass.setter
    def airmass(self, value):
        self._airmass = hp.ud_grade(value, nside_out=self.nside)

    @property
    def altAz(self):
        if self._azAlt is None:
            self.calc_altAz()
        return self._altAz

    def calc_altAz(self):
        alt, az = _approx_RaDec2AltAz(self.ra_rad, self.dec_rad,
                                      self.site.latitude_rad,
                                      self.longitude_rad, self._mjd)
        self._altAz = {'alt_rad': alt, 'az_rad': az}

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        # Set things that need to be recalculated to None
        self._azAlt = None

    @property
    def skybrightness(self):
        return self._skybrightness

    @skybrightness.setter
    def skybrightness(self, indict):
        for key in indict:
            self._skybrightness[key] = hp.ud_grade(indict[key], nside_out=self.nside)
        # If sky brightness changes, need to recalc M5 depth.
        self._M5Depth = None

    @property
    def M5Depth(self):
        if self._M5Depth is None:
            self.calc_M5Depth()
        return self._M5Depth

    def calc_M5Depth(self):
        self._M5Depth = {}
        for filtername in self._skybrightness:
            good = np.where(self._skybrightness[filtername] != hp.UNSEEN)
            self._M5Depth[filtername] = self.zeros_map.copy().fill(hp.UNSEEN)
            self._M5Depth[filtername][good] = m5_flat_sed(self.filtername,
                                                          self._skybrightness[filtername][good],
                                                          self._FWHMeff[filtername][good],
                                                          self.expTime,
                                                          self._airmass[good])
            self._M5Depth[filtername] = ma.masked_values(self._M5Depth[filtername], hp.UNSEEN)


