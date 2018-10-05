import numpy as np
from lsst.sims.utils import _approx_RaDec2AltAz, Site, _hpid2RaDec, m5_flat_sed
import healpy as hp
import numpy.ma as ma


__all__ = ['Conditions']


class Conditions(object):
    """
    Class to hold telemetry information

    If the incoming value is a healpix map, we use a setter to ensure the
    resolution matches.
    """
    def __init__(self, nside, site='LSST', expTime=30.):
        """
        Parameters
        ----------
        expTime : float (30)
            The exposure time to assume when computing the 5-sigma limiting depth

        All angles stored as radians. LMST as hours.

        Attributes are all one of:
        * Float or int
        * healpix map
        * dicts of healpix maps keyed by filtername
        """

        self.nside = nside
        self.site = Site(site)
        self.expTime = expTime
        hpids = np.arange(hp.nside2npix(nside))
        # Generate an empty map so we can copy when we need a new map
        self.zeros_map = np.zeros(hp.nside2npix(nside), dtype=float)
        # The RA, Dec grid we are using
        self.ra, self.dec = _hpid2RaDec(nside, hpids)

        # Modified Julian Date (day)
        self._mjd = None
        # Altitude and azimuth. Dict with degrees and radians
        self._alt = None
        self._az = None
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
        self._airmass = None

        # Attribute to hold the current observing queue
        self.queue = None

        self._HP2Fields = None

        # Moon
        self.moonAlt = None
        self.moonAz = None
        self.moonRA = None
        self.moonDec = None
        self.moonPhase = None

        # Sun
        self.sunAlt = None
        self.sunAz = None

        self.night = None
        self.last_twilight_end = None
        self.next_twilight_start = None

        # Current telescope pointing
        self.telRA = None
        self.telDec = None

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
        self._M5Depth = None

    @property
    def alt(self):
        if self._alt is None:
            self.calc_altAz()
        return self._altAz

    @property
    def az(self):
        if self._az is None:
            self.calc_altAz()
        return self._az

    def calc_altAz(self):
        self._alt, self._az = _approx_RaDec2AltAz(self.ra, self.dec,
                                                  self.site.latitude_rad,
                                                  self.site.longitude_rad, self._mjd)
    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        # Set things that need to be recalculated to None
        self._az = None
        self._alt = None

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
    def FWHMeff(self):
        return self._FWHMeff

    @FWHMeff.setter
    def FWHMeff(self, indict):
        for key in indict:
            self._FWHMeff[key] = indict[key]
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

    @property
    def HP2Fields(self):
        # XXX--not sure what this one is
        return self._HP2Fields

    

