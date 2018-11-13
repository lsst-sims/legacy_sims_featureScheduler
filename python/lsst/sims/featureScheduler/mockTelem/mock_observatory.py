import numpy as np
from lsst.sims.utils import (_hpid2RaDec, _raDec2Hpid, Site, calcLmstLast,
                             m5_flat_sed, _approx_RaDec2AltAz, _angularSeparation)
import lsst.sims.skybrightness_pre as sb
import healpy as hp
from datetime import datetime
from lsst.sims.downtimeModel import ScheduledDowntime, UnscheduledDowntime
from lsst.sims.seeingModel import SeeingSim
from lsst.sims.cloudModel import CloudModel
from lsst.sims.featureScheduler.features import Conditions
from lsst.sims.featureScheduler.utils import set_default_nside
from lsst.ts.observatory.model import ObservatoryModel, Target
from astropy.coordinates import get_sun, get_moon, EarthLocation, AltAz
from astropy.time import Time
from lsst.utils import getPackageDir
import os


__all__ = ['Mock_observatory']


class ExtendedObservatoryModel(ObservatoryModel):
    """Add some functionality to ObservatoryModel
    """

    def expose(self, target):
        # Break out the exposure command from observe method
        visit_time = sum(target.exp_times) + \
            target.num_exp * self.params.shuttertime + \
            max(target.num_exp - 1, 0) * self.params.readouttime
        self.update_state(self.current_state.time + visit_time)

    def observe_times(self, target):
        """observe a target, and return the slewtime and visit time for the action
        Note, slew and expose will update the current_state
        """
        t1 = self.current_state.time + 0
        # Note, this slew assumes there is a readout that needs to be done.
        self.slew(target)
        t2 = self.current_state.time + 0
        self.expose(target)
        t3 = self.current_state.time + 0
        slewtime = t2 - t1
        visitime = t3 - t2
        return slewtime, visitime


class dummy_time_handler(object):
    """
    Don't need the full time handler, so save a dependency and use this.
    """
    def __init__(self, mjd_init):
        """
        Parameters
        ----------
        mjd_init : float
            The initial mjd
        """
        self._unix_start = datetime(1970, 1, 1)
        t = Time(mjd_init, format='mjd')
        self.initial_dt = t.datetime

    def time_since_given_datetime(self, datetime1, datetime2=None, reverse=False):
        """
        Really? We need a method to do one line of arithmatic?
        """
        if datetime2 is None:
            datetime2 = self._unix_start
        if reverse:
            return (datetime1 - datetime2).total_seconds()
        else:
            return (datetime2 - datetime1).total_seconds()


class filter_swap_scheduler(object):
    """A simple way to schedule what filter to load
    """
    def __init__(self):
        pass

    def __call__(self, conditions):
        # Just based on lunar illumination, decide if some different filter should be loaded.
        pass


class Mock_observatory(object):
    """A class to generate a realistic telemetry stream for the scheduler
    """

    def __init__(self, nside=None, mjd_start=59853.5, seed=42, quickTest=True,
                 alt_min=5., lax_dome=True, cloud_limit=0.699):
        """
        Parameters
        ----------
        nside : int (None)
            The healpix nside resolution
        mjd_start : float (59853.5)
            The MJD to start the observatory up at
        alt_min : float (5.)
            The minimum altitude to compute models at (degrees).
        lax_dome : bool (True)
            Passed to observatory model. If true, allows dome creep.
        cloud_limit : float (0.7)
            The limit to stop taking observations if the cloud model returns something equal or higher
        """

        if nside is None:
            nside = set_default_nside()
        self.nside = nside

        self.cloud_limit = cloud_limit

        self.alt_min = np.radians(alt_min)
        self.lax_dome = lax_dome

        self.mjd_start = mjd_start
        

        # Conditions object to update and return on request
        self.conditions = Conditions(nside=self.nside)

        # Create an astropy location
        self.site = Site('LSST')
        self.location = EarthLocation(lat=self.site.latitude, lon=self.site.longitude,
                                      height=self.site.height)

        # Load up all the models we need
        # Make my dummy time handler
        dth = dummy_time_handler(mjd_start)
        
        # Downtime
        self.down_nights = []
        sdt = ScheduledDowntime()
        sdt.initialize()
        usdt = UnscheduledDowntime()
        usdt.initialize(random_seed=seed)

        for downtime in sdt.downtimes:
            self.down_nights.extend(range(downtime[0], downtime[0]+downtime[1], 1))
        for downtime in usdt.downtimes:
            self.down_nights.extend(range(downtime[0], downtime[0]+downtime[1], 1))
        self.down_nights.sort()

        self.seeing_model = SeeingSim(dth)

        self.cloud_model = CloudModel(dth)
        self.cloud_model.read_data()

        self.sky_model = sb.SkyModelPre(speedLoad=quickTest)

        self.observatory = ExtendedObservatoryModel()
        self.observatory.configure_from_module()
        # Make it so it respects my requested rotator angles
        self.observatory.params.rotator_followsky = True

        self.filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
        self.seeing_FWHMeff = {}
        for key in self.filterlist:
            self.seeing_FWHMeff[key] = np.zeros(hp.nside2npix(self.nside), dtype=float)

        self._load_almanac()

        # Let's make sure we're at an openable MJD
        good_mjd = False
        to_set_mjd = mjd_start
        while not good_mjd:
            good_mjd, to_set_mjd = self.check_mjd(to_set_mjd)
        self.mjd = to_set_mjd

    def _load_almanac(self):
        file = os.path.join(getPackageDir('sims_featureScheduler'),
                            'python/lsst/sims/featureScheduler/mockTelem/almanac.npz')
        temp = np.load(file)
        self.almanac = temp['almanac'].copy()
        temp.close()
        # Set the night index based on the starting MJD
        loc = np.searchsorted(self.almanac['sunset'], self.mjd_start)
        # Set the start MJD to be night 1.
        self.almanac['night'] -= self.almanac['night'][loc-1]

    def return_conditions(self):
        """

        Returns
        -------
        lsst.sims.featureScheduler.features.conditions object
        """

        self.conditions.mjd = self.mjd

        self.conditions.night = self.night
        # Time since start of simulation
        delta_t = (self.mjd-self.mjd_start)*24.*3600.

        # Clouds
        self.conditions.clouds = self.cloud_model.get_cloud(delta_t)

        # use conditions object itself to get aprox altitude of each healpx
        alts = self.conditions.alt
        azs = self.conditions.az

        good = np.where(alts > self.alt_min)

        # Compute the airmass at each heapix
        airmass = np.zeros(alts.size, dtype=float)
        airmass.fill(np.nan)
        airmass[good] = 1./np.cos(np.pi/2. - alts[good])
        self.conditions.airmass = airmass

        # reset the seeing
        for key in self.seeing_FWHMeff:
            self.seeing_FWHMeff[key].fill(np.nan)
        # Use the model to get the seeing at this time and airmasses.
        fwhm_500, fwhm_eff, fwhm_geom = self.seeing_model.get_seeing(delta_t, airmass[good])
        for i, key in enumerate(self.seeing_model.filter_list):
            self.seeing_FWHMeff[key][good] = fwhm_eff[i, :]
        self.conditions.FWHMeff = self.seeing_FWHMeff

        # sky brightness
        self.conditions.skybrightness = self.sky_model.returnMags(self.mjd)

        self.conditions.mounted_filters = self.observatory.current_state.mountedfilters
        self.conditions.curret_filter = self.observatory.current_state.filter

        # Compute the slewtimes
        slewtimes = np.empty(alts.size, dtype=float)
        slewtimes.fill(np.nan)
        slewtimes[good] = self.observatory.get_approximate_slew_delay(alts[good], azs[good],
                                                                      self.observatory.current_state.filter,
                                                                      lax_dome=self.lax_dome)
        # Mask out anything the slewtime says is out of bounds
        slewtimes[np.where(slewtimes < 0)] = np.nan
        self.conditions.slewtime = slewtimes

        # Let's get the sun and moon
        t = Time(self.mjd, format='mjd', location=self.location)
        sun = get_sun(t)
        moon = get_moon(t)

        # Using fast alt,az
        sunAlt, sunAz = _approx_RaDec2AltAz(np.array([sun.ra.rad]), np.array([sun.dec.rad]),
                                            self.location.lat.rad,
                                            self.location.lon.rad,
                                            self.mjd)

        moonAlt, moonAz = _approx_RaDec2AltAz(np.array([moon.ra.rad]), np.array([moon.dec.rad]),
                                              self.location.lat.rad,
                                              self.location.lon.rad,
                                              self.mjd)

        moon_sun_sep = _angularSeparation(sun.ra.rad, sun.dec.rad, moon.ra.rad, moon.dec.rad)
        self.conditions.moonPhase = np.max(moon_sun_sep/np.pi*100.)

        self.conditions.moonAlt = moonAlt.min()
        self.conditions.moonAz = moonAz.min()
        self.conditions.moonRA = moon.ra.rad
        self.conditions.moonDec = moon.dec.rad

        self.conditions.sunAlt = sunAlt.min()

        self.conditions.lmst, last = calcLmstLast(self.mjd, self.site.longitude_rad)

        self.conditions.telRA = self.observatory.current_state.ra_rad
        self.conditions.telDec = self.observatory.current_state.dec_rad
        self.conditions.telRotSkyPos = self.observatory.current_state.ang_rad

        # To set

        # conditions.last_twilight_end
        # conditions.next_twilight_start

        return self.conditions

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = np.searchsorted(self.almanac['sunset'], value)
        self.night = self.almanac['night'][self.almanac_indx]

    def observation_add_data(self, observation):
        """
        Fill in the metadata for a completed observation
        """
        # Time since start of simulation
        delta_t = (self.mjd-self.mjd_start)*24.*3600.
        # XXX--todo

        return observation

    def check_mjd(self, mjd):
        """See if an mjd is ok to observe

        Returns
        -------
        bool

        mdj : float
            If True, the input mjd. If false, a good mjd to skip forward to.
        """
        # check the clouds
        delta_t = (mjd-self.mjd_start)*24.*3600.
        clouds = self.cloud_model.get_cloud(delta_t)
        if clouds > self.cloud_limit:
            # Let's just reach into the cloud model and see when it's not cloudy anymore
            jump_to = np.where((self.cloud_model.cloud_dates > delta_t) &
                               (self.cloud_model.cloud_values < self.cloud_limit))[0].min()

            return False, self.mjd_start + self.cloud_model.dates[jump_to]/24./3600.
        alm_indx = np.searchsorted(self.almanac['sunset'], mjd, side='right')
        # at the end of the night, advance to the next setting twilight
        if mjd > self.almanac['sun_n12_rising'][alm_indx]:
            return False, self.almanac['sun_n12_setting'][alm_indx+1]
        # We're in a down night, advance to next night
        if self.almanac['night'][alm_indx] in self.down_nights:
            return False, self.almanac['sun_n12_setting'][alm_indx+1]
        return True, mjd

    def observe(self, observation):
        """Try to make an observation

        Returns
        -------
        status : bool
            Result of if the observation worked
        observation : observation object
            None if there was no observation taken. Completed observation with meta data filled in.
        new_night : bool
            Have we started a new night.
        """

        start_night = self.night.copy()
        # slew to the target--note that one can't slew without also incurring a readtime penalty?
        target = Target(band_filter=observation['filter'], ra_rad=observation['RA'],
                        dec_rad=observation['dec'], ang_rad=observation['rotSkyPos'],
                        num_exp=observation['nexp'], exp_times=[observation['exptime']])
        slewtime, visittime = self.observatory.observe_times(target)

        # Check if the mjd after slewtime and visitime is fine:
        observation_worked, new_mjd = self.check_mjd(self.mjd + slewtime + visittime)
        if observation_worked:
            observation['visittime'] = visittime
            observation['slewtime'] = slewtime
            self.mjd = self.mjd + slewtime
            # Metadata on observation is after slew and settle, so at start of exposure.
            result = self.observation_data(observation)
            self.mjd = self.mjd + visittime
            new_night = False
        else:
            result = None
            self.observatory.park()
            # Skip to next legitimate mjd
            self.mjd = new_mjd
            now_night = self.night
            new_night = now_night != start_night

        return observation_worked, result, new_night
