import numpy as np
from lsst.sims.utils import haversine, _hpid2RaDec
import lsst.sims.skybrightness_pre as sb
import healpy as hp
import lsst.sims.featureScheduler.utils as utils

sec2days = 1./(3600.*24.)
default_nside = utils.set_default_nside()


class Speed_observatory(object):
    """
    A very very simple observatory model that will take observation requests and supply
    current conditions.
    """
    def __init__(self, mjd_start=59580.035, ang_speed=0.5,
                 readtime=2., settle=2., filtername=None, f_change_time=120.,
                 nside=default_nside, sun_limit=-12.):
        """
        Parameters
        ----------
        mjd_start : float (59580.035)
            The Modified Julian Date to set the observatory to.
        ang_speed : float (10.)
            The angular speed the telescope can slew at in degrees per second.
        readtime : float (2.)
            The time it takes to read out the camera (seconds).
        settle : float (2.)
            The time it takes the telescope to settle after slewing (seconds)
        filtername : str (None)
            The filter to start the observatory loaded with
        f_change_time : float (120.)
            The time it takes to change filters (seconds)
        nside : int (32)
            The healpixel nside to make sky calculations on.
        sun_limit : float (-12.)
            The altitude limit for the sun (degrees)
        """
        self.mjd = mjd_start
        self.ang_speed = np.radians(ang_speed)
        self.settle = settle
        self.f_change_time = f_change_time
        self.readtime = readtime
        self.sun_limit = np.radians(sun_limit)
        # Load up the sky brightness model
        self.sky = sb.SkyModelPre(preload=False)

        # Start out parked
        self.ra = None
        self.dec = None
        self.filtername = None

        # Set up all sky coordinates
        hpids = np.arange(hp.nside2npix(nside))
        self.ra_all_sky, self.dec_all_sky = _hpid2RaDec(nside, hpids)

    def slew_time(self, ra, dec):
        """
        Compute slew time to new ra, dec position
        """
        dist = haversine(ra, dec, self.ra, self.dec)
        time = dist / self.ang_speed
        return time

    def return_status(self):
        """
        Return a dict full of the current info about the observatory and sky.
        """
        result = {}
        result['mjd'] = self.mjd
        result['sky_brightness'] = self.sky.returnMags(self.mjd)
        # XXX Obviously need to update to a real seeing table
        result['seeing'] = 0.7  # arcsec
        result['airmass'] = self.sky.returnAirmass(self.mjd)

        return result

    def check_mjd(self, mjd):
        """
        If an mjd is not in daytime or downtime
        """
        sunMoon = self.sky.returnSunMoon(mjd)
        if sunMoon['sunAlt'] > self.sun_limit:
            good = np.where(self.sky.info['mjds'] > mjd & (self.sky.info['sunAlts'] > self.sun_limit))[0]
            mjd = np.min(self.sky.info['mjds'][good])
            return False, mjd
        else:
            return True, mjd

    def attempt_observe(self, observation):
        """
        Check an observation, if there is enough time, execute it and return it, otherwise, return none.
        """
        # If we were in a parked position, assume no time lost to slew, settle, filter change
        if self.ra is not None:
            st = self.slew_time(observation['ra'], observation['dec'])
            self.filtername = observation['filter']
            settle = self.settle
            if self.filtername != observation['filter']:
                ft = self.f_change_time
            else:
                ft = 0.
        else:
            st = 0.
            settle = 0.
            ft = 0.

        # Assume we can slew while reading the last exposure, and slewtime always > exptime
        rt = (observation['nexp']-1.)*self.readtime
        total_time = st + rt + observation['exptime'] + settle

        check_result, jump_mjd = self.check_mjd(self.mjd + total_time)
        if check_result:
            # time the shutter should open
            observation['mjd'] = self.mjd + (st + ft + self.settle) * sec2days
            self.mjd += total_time*sec2days
            self.ra = observation['ra']
            self.dec = observation['dec']
            self.filtername = observation['filter']
            return observation
        else:
            self.mjd = jump_mjd
            self.ra = None
            self.dec = None
            return None

