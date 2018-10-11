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
    def __init__(self, nside, site='LSST', exptime=30.):
        """
        Parameters
        ----------
        nside : int
            The healpixel nside to set the resolution of attributes.
        site : str ('LSST')
            A site name used to create a sims.utils.Site object. For looking up
            observatory paramteres like latitude and longitude.
        expTime : float (30)
            The exposure time to assume when computing the 5-sigma limiting depth

        Attributes
        ----------
        nside : int
            Healpix resolution
        site : lsst.sims.Site object
            Contains static site-specific data (lat, lon, altitude, etc)
        ra : np.array
            A healpix array with the RA of each healpixel center (radians).
        dec : np.array
            A healpix array with the Dec of each healpixel center (radians).
        mjd : float
            Modified Julian Date (days)
        alt : np.array
            Altitude of each healpixel (radians). Recaclulated if mjd is updated.
        az : np.array
            Azimuth of each healpixel (radians). Recaclulated if mjd is updated.
        clouds : float
            The fraction of sky covered by clouds. (In the future might update to transparency map)
        slewtime : np.array
            Healpix showing the slewtime to each healpixel center (seconds)
        current_filter : str
            The name of the current filter. (expect one of u, g, r, i, z, y).
        mounted_filters : list of str
            The filters that are currently mounted and thu available (expect 5 of u, g, r, i, z, y)
        night : int
            The current night number (days). Probably starts at 1.
        lmst : float
            The local mean sidearal time (hours). Updates is mjd is changed.
        skybrightness : dict of np.array
            Dictionary keyed by filtername. Values are healpix arrays with the sky brightness at each
            healpix center (mag/acsec^2)
        FWHMeff : dict of np.array
            Dictionary keyed by filtername. Values are the effective seeing FWHM at each healpix
            center (arcseconds)
        M5Depth : 
        queue : 
        moonAlt
        moonAz
        moonRA
        moonDec
        moonPhase
        sunAlt
        sunAz
        last_twilight_end
        next_twilight_start
        telRA
        telDec
        cloud_map
        HA
        """

        self.nside = nside
        self.site = Site(site)
        self.exptime = exptime
        hpids = np.arange(hp.nside2npix(nside))
        # Generate an empty map so we can copy when we need a new map
        self.zeros_map = np.zeros(hp.nside2npix(nside), dtype=float)
        self.unseen_map = np.zeros(hp.nside2npix(nside), dtype=float)
        self.unseen_map.fill(hp.UNSEEN)
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
        self._lmst = None
        # Should be a dict with filtername keys
        self._skybrightness = {}
        self._FWHMeff = {}
        self._M5Depth = None
        self._airmass = None

        # Attribute to hold the current observing queue
        self.queue = None

        # Moon
        self.moonAlt = None
        self.moonAz = None
        self.moonRA = None
        self.moonDec = None
        self.moonPhase = None

        # Sun
        self.sunAlt = None
        self.sunAz = None

        self.last_twilight_end = None
        self.next_twilight_start = None

        # Current telescope pointing
        self.telRA = None
        self.telDec = None

        # Full sky cloud map
        self._cloud_map = None
        self._HA = None

    @property
    def lmst(self):
        return self._lmst
    @lmst.setter
    def lmst(self, value):
        self._lmst = value
        self._HA = None

    @property
    def HA(self):
        if self._HA is None:
            self.calc_HA()
        return self._HA
    
    def calc_HA(self):
        self._HA = np.radians(self._lmst*360./24.) - self.ra
        self._HA[np.where(self._HA < 0)] += 2.*np.pi

    @property
    def cloud_map(self):
        return self._cloud_map

    @cloud_map.setter
    def cloud_map(self, value):
        self._cloud_map = hp.ud_grade(value, nside_out=self.nside)

    @property
    def slewtime(self):
        return self._slewtime

    @slewtime.setter
    def slewtime(self, value):
        # Using 0 for start of night
        if np.size(value) == 1:
            self._slewtime = value
        else:
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
        return self._alt

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
            self._FWHMeff[key] = hp.ud_grade(indict[key], nside_out=self.nside)
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
            self._M5Depth[filtername] = self.unseen_map.copy()
            self._M5Depth[filtername][good] = m5_flat_sed(filtername,
                                                          self._skybrightness[filtername][good],
                                                          self._FWHMeff[filtername][good],
                                                          self.exptime,
                                                          self._airmass[good])

            self._M5Depth[filtername] = ma.masked_values(self._M5Depth[filtername], hp.UNSEEN)
