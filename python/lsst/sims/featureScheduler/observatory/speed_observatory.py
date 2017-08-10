from builtins import zip
from builtins import object
import numpy as np
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, calcLmstLast
import lsst.sims.skybrightness_pre as sb
import healpy as hp
import lsst.sims.featureScheduler.utils as utils
import ephem
from slew_pre import Slewtime_pre
from lsst.sims.utils import m5_flat_sed


__all__ = ['Speed_observatory']


sec2days = 1./(3600.*24.)
default_nside = utils.set_default_nside()
doff = ephem.Date(0)-ephem.Date('1858/11/17')


class Speed_observatory(object):
    """
    A very very simple observatory model that will take observation requests and supply
    current conditions.
    """
    def __init__(self, mjd_start=59580.035,
                 readtime=2., filtername=None, f_change_time=140.,
                 nside=default_nside, sun_limit=-13., quickTest=True, alt_limit=20.):
        """
        Parameters
        ----------
        mjd_start : float (59580.035)
            The Modified Julian Date to set the observatory to.
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
        self.f_change_time = f_change_time
        self.readtime = readtime
        self.sun_limit = np.radians(sun_limit)
        self.alt_limit = np.radians(alt_limit)
        # Load up the sky brightness model
        self.sky = sb.SkyModelPre(preload=False, speedLoad=quickTest)
        # Should realy set this by inspecting the map.
        self.sky_nside = 32

        # Start out parked
        self.ra = None
        self.dec = None
        self.filtername = None

        # Set up all sky coordinates
        hpids = np.arange(hp.nside2npix(self.sky_nside))
        self.ra_all_sky, self.dec_all_sky = _hpid2RaDec(self.sky_nside, hpids)
        self.status = None

        self.site = Site(name='LSST')
        self.obs = ephem.Observer()
        self.obs.lat = self.site.latitude_rad
        self.obs.lon = self.site.longitude_rad
        self.obs.elevation = self.site.height

        self.obs.horizon = 0.

        self.sun = ephem.Sun()

        # Generate sunset times so we can label nights by integers
        self.generate_sunsets()
        self.night = self.mjd2night(self.mjd)

        # Make a slewtime interpolator
        self.slew_interp = Slewtime_pre()

    def slew_time(self, alt, az, mintime=2.):
        """
        Compute slew time to new ra, dec position
        """

        current_alt, current_az = utils.stupidFast_RaDec2AltAz(np.array([self.ra]),
                                                               np.array([self.dec]),
                                                               self.obs.lat, self.obs.lon,
                                                               self.mjd)
        # Interpolation can be off by ~.1 seconds if there's no slew.
        if (np.max(current_alt) == np.max(alt)) & (np.max(current_az) == np.max(az)):
            time = mintime
        else:
            time = self.slew_interp(current_alt, current_az, alt, az)
        return time

    def slewtime_map(self):
        """
        Return a map of how long it would take to slew to lots of positions
        """
        if self.ra is None:
            return 0.
        alt, az = utils.stupidFast_RaDec2AltAz(self.ra_all_sky, self.dec_all_sky,
                                               self.obs.lat, self.obs.lon, self.mjd)
        good = np.where(alt >= self.alt_limit)
        result = np.empty(self.ra_all_sky.size, dtype=float)
        result.fill(hp.UNSEEN)
        result[good] = self.slew_time(alt[good], az[good])
        return result

    def return_status(self):
        """
        Return a dict full of the current info about the observatory and sky.

        XXX-- Need to document all these with units!!!
        """
        result = {}
        result['mjd'] = self.mjd
        result['night'] = self.night
        result['lmst'], last = calcLmstLast(self.mjd, self.site.longitude_rad)
        result['skybrightness'] = self.sky.returnMags(self.mjd)
        result['slewtimes'] = self.slewtime_map()
        result['airmass'] = self.sky.returnAirmass(self.mjd)
        # XXX Obviously need to update to a real seeing table, and make it a full-sky map, and filter, airmass dependent
        result['FWHMeff'] = np.empty(result['airmass'].size)  # arcsec
        result['FWHMeff'].fill(0.7)
        result['filter'] = self.filtername
        result['RA'] = self.ra
        result['dec'] = self.dec
        result['next_twilight_start'] = self.next_twilight_start(self.mjd)
        result['next_twilight_end'] = self.next_twilight_end(self.mjd)
        result['last_twilight_end'] = self.last_twilight_end(self.mjd)
        self.status = result
        return result

    def check_mjd(self, mjd):
        """
        If an mjd is not in daytime or downtime
        """
        sunMoon = self.sky.returnSunMoon(mjd)
        if sunMoon['sunAlt'] > self.sun_limit:
            good = np.where((self.sky.info['mjds'] >= mjd) & (self.sky.info['sunAlts'] <= self.sun_limit))[0]
            if np.size(good) == 0:
                # hack to advance if we are at the end of the mjd list I think
                mjd += 0.25
            else:
                mjd = self.sky.info['mjds'][good][0]
            return False, mjd
        else:
            return True, mjd

    def attempt_observe(self, observation_in, indx=None):
        """
        Check an observation, if there is enough time, execute it and return it, otherwise, return none.
        """
        # If we were in a parked position, assume no time lost to slew, settle, filter change

        observation = observation_in.copy()
        alt, az = utils.stupidFast_RaDec2AltAz(np.array([observation['RA']]),
                                               np.array([observation['dec']]),
                                               self.obs.lat, self.obs.lon, self.mjd)
        if self.ra is not None:
            if self.filtername != observation['filter']:
                ft = self.f_change_time
                st = 0.
            else:
                ft = 0.
                st = self.slew_time(alt, az)
        else:
            st = 0.
            ft = 0.

        # Assume we can slew while reading the last exposure
        # So, filter change time, slew to target time, expose time, read time
        rt = (observation['nexp']-1.)*self.readtime
        total_time = (ft + st + observation['exptime'] + rt)*sec2days
        check_result, jump_mjd = self.check_mjd(self.mjd + total_time)
        if check_result:
            # XXX--major decision here, should the status be updated after every observation? Or just assume
            # airmass, seeing, and skybrightness do not change significantly?
            if self.ra is None:
                update_status = True
            else:
                update_status = False
            # This should be the start of the exposure.
            observation['mjd'] = self.mjd + (ft + st)*sec2days
            observation['night'] = self.night
            # XXX I REALLY HATE THIS! READTIME SHOULD NOT BE LUMPED IN WITH SLEWTIME!
            observation['slewtime'] = ft+st+rt
            self.mjd = self.mjd + total_time
            self.night = self.mjd2night(self.mjd)
            self.ra = observation['RA']
            self.dec = observation['dec']
            if update_status:
                # What's the name for temp variables?
                status = self.return_status()

            self.filtername = observation['filter'][0]
            hpid = _raDec2Hpid(self.sky_nside, self.ra, self.dec)
            observation['skybrightness'] = self.sky.returnMags(observation['mjd'], indx=hpid,
                                                               extrapolate=True)[self.filtername]
            observation['FWHMeff'] = self.status['FWHMeff'][hpid]
            observation['airmass'] = self.status['airmass'][hpid]
            observation['fivesigmadepth'] = m5_flat_sed(observation['filter'][0],
                                                        observation['skybrightness'],
                                                        observation['FWHMeff'],
                                                        observation['exptime'],
                                                        observation['airmass'])
            observation['alt'] = alt
            observation['az'] = az
            return observation
        else:
            self.mjd = jump_mjd
            self.night = self.mjd2night(self.mjd)
            self.ra = None
            self.dec = None
            self.status = None
            self.filtername = None
            return None

    def generate_sunsets(self, nyears=13, day_pad=50):
        """
        Generate the sunrise times for LSST so we can label nights by MJD
        """

        # Set observatory horizon to zero
        self.obs.horizon = 0.

        # Swipe dates to match sims_skybrightness_pre365
        mjd_start = self.mjd
        mjd_end = np.arange(59560, 59560+365.25*nyears+day_pad+366, 366).max()
        step = 0.25
        mjds = np.arange(mjd_start, mjd_end+step, step)
        setting = mjds*0.

        # Stupid Dublin Julian Date
        djds = mjds - doff
        sun = ephem.Sun()

        for i, (mjd, djd) in enumerate(zip(mjds, djds)):
            sun.compute(djd)
            setting[i] = self.obs.previous_setting(sun, start=djd, use_center=True)
        setting = setting + doff

        # zomg, round off crazy floating point precision issues
        setting_rough = np.round(setting*100.)
        u, indx = np.unique(setting_rough, return_index=True)
        self.setting_sun_mjds = setting[indx]
        left = np.searchsorted(self.setting_sun_mjds, mjd_start)
        self.setting_sun_mjds = self.setting_sun_mjds[left:]

    def next_twilight_start(self, mjd, twi_limit=-18.):
        # find the next rising twilight. String to make it degrees I guess?
        self.obs.horizon = str(twi_limit)
        next_twi = self.obs.next_rising(self.sun, start=mjd-doff)+doff
        return next_twi

    def next_twilight_end(self, mjd, twi_limit=-18.):
        self.obs.horizon = str(twi_limit)
        next_twi = self.obs.next_setting(self.sun, start=mjd-doff)+doff
        return next_twi

    def last_twilight_end(self, mjd, twi_limit=-18.):
        self.obs.horizon = str(twi_limit)
        next_twi = self.obs.previous_setting(self.sun, start=mjd-doff)+doff
        return next_twi

    def mjd2night(self, mjd):
        """
        Convert an mjd to a night integer.
        """
        return np.searchsorted(self.setting_sun_mjds, mjd)

    def set_mjd(self, mjd):
        """
        update the mjd of the observatory
        """
        self.mjd = mjd
        self.night = self.mjd2night(mjd)






