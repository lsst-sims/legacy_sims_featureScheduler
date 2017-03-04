import numpy as np
from lsst.sims.utils import haversine, _hpid2RaDec, _raDec2Hpid
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
    def __init__(self, mjd_start=59580.035, ang_speed=5.,
                 readtime=2., settle=2., filtername=None, f_change_time=120.,
                 nside=default_nside, sun_limit=-13., quickTest=True):
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
        quickTest : bool (True)
            Load only a small pre-computed sky array rather than a full year.
        """
        self.mjd = mjd_start
        self.ang_speed = np.radians(ang_speed)
        self.settle = settle
        self.f_change_time = f_change_time
        self.readtime = readtime
        self.sun_limit = np.radians(sun_limit)
        # Load up the sky brightness model
        self.sky = sb.SkyModelPre(preload=False, speedLoad=quickTest)
        # Should realy set this by inspecting the map.
        self.sky_nside = 32

        # Start out parked
        self.ra = None
        self.dec = None
        self.filtername = None

        # Set up all sky coordinates
        hpids = np.arange(hp.nside2npix(nside))
        self.ra_all_sky, self.dec_all_sky = _hpid2RaDec(nside, hpids)
        self.status = None

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
        result['skybrightness'] = self.sky.returnMags(self.mjd)
        result['airmass'] = self.sky.returnAirmass(self.mjd)
        # XXX Obviously need to update to a real seeing table, and make it a full-sky map, and filter, airmass dependent
        result['FWHMeff'] = np.empty(result['airmass'].size)  # arcsec
        result['FWHMeff'].fill(0.7)
        result['filter'] = self.filtername
        result['RA'] = self.ra
        result['dec'] = self.dec

        self.status = result
        return result

    def check_mjd(self, mjd):
        """
        If an mjd is not in daytime or downtime
        """
        sunMoon = self.sky.returnSunMoon(mjd)
        if sunMoon['sunAlt'] > self.sun_limit:
            good = np.where((self.sky.info['mjds'] > mjd) & (self.sky.info['sunAlts'] < self.sun_limit))[0]
            mjd = self.sky.info['mjds'][good][0]
            return False, mjd
        else:
            return True, mjd

    def attempt_observe(self, observation, indx=None):
        """
        Check an observation, if there is enough time, execute it and return it, otherwise, return none.
        """
        # If we were in a parked position, assume no time lost to slew, settle, filter change
        if self.ra is not None:
            st = self.slew_time(observation['RA'], observation['dec'])
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
        total_time = (st + rt + observation['exptime'] + settle + ft)*sec2days
        to_open_time = (st+settle+ft)*sec2days
        check_result, jump_mjd = self.check_mjd(self.mjd + total_time)
        if check_result:
            # XXX--major decision here, should the status be updated after every observation? Or just assume
            # airmass, seeing, and skybrightness do not change significantly?
            if self.ra is None:
                update_status = True
            else:
                update_status = False
            self.mjd = self.mjd+to_open_time
            observation['mjd'] = self.mjd
            self.ra = observation['RA']
            self.dec = observation['dec']
            if update_status:
                # What's the name for temp variables?
                status = self.return_status()
            # time the shutter should open
            self.mjd += total_time-to_open_time

            self.filtername = observation['filter'][0]
            hpid = _raDec2Hpid(self.sky_nside, self.ra, self.dec)
            observation['skybrightness'] = self.status['skybrightness'][self.filtername][hpid]
            observation['FWHMeff'] = self.status['FWHMeff'][hpid]
            observation['airmass'] = self.status['airmass'][hpid]
            return observation
        else:
            self.mjd = jump_mjd
            self.ra = None
            self.dec = None
            self.status = None
            self.filtername = None
            return None

