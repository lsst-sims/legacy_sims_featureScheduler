import numpy as np
from lsst.sims.utils import (_hpid2RaDec, _raDec2Hpid, Site, calcLmstLast,
                             m5_flat_sed, _approx_RaDec2AltAz, _angularSeparation)
import lsst.sims.skybrightness_pre as sb
import healpy as hp
from datetime import datetime
from lsst.sims.downtimeModel import ScheduledDowntime, UnscheduledDowntime
import lsst.sims.downtimeModel as downtimeModel
from lsst.sims.seeingModel import SeeingSim
from lsst.sims.cloudModel import CloudModel
from lsst.sims.featureScheduler.features import Conditions
from lsst.sims.featureScheduler.utils import set_default_nside
from lsst.ts.observatory.model import ObservatoryModel, Target
from astropy.coordinates import EarthLocation
from astropy.time import Time
from lsst.sims.almanac import Almanac
import warnings
import matplotlib.pylab as plt
from lsst.ts.observatory.model import ObservatoryState
from importlib import import_module

__all__ = ['Model_observatory']


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
        if not self.current_state.tracking:
            ValueError('Telescope model stopped tracking, that seems bad.')
        slewtime = t2 - t1
        visitime = t3 - t2
        return slewtime, visitime

    #  Adding wrap_padding to make azimuth slews more intelligent
    def get_closest_angle_distance(self, target_rad, current_abs_rad,
                                   min_abs_rad=None, max_abs_rad=None,
                                   wrap_padding=0.873):
        """Calculate the closest angular distance including handling \
           cable wrap if necessary.

        Parameters
        ----------
        target_rad : float
            The destination angle (radians).
        current_abs_rad : float
            The current angle (radians).
        min_abs_rad : float, optional
            The minimum constraint angle (radians).
        max_abs_rad : float, optional
            The maximum constraint angle (radians).
        wrap_padding : float (0.873)
            The amount of padding to use to make sure we don't track into limits (radians).


        Returns
        -------
        tuple(float, float)
            (accumulated angle in radians, distance angle in radians)
        """
        # if there are wrap limits, normalizes the target angle
        TWOPI = 2 * np.pi
        if min_abs_rad is not None:
            norm_target_rad = divmod(target_rad - min_abs_rad, TWOPI)[1] + min_abs_rad
            if max_abs_rad is not None:
                # if the target angle is unreachable
                # then sets an arbitrary value
                if norm_target_rad > max_abs_rad:
                    norm_target_rad = max(min_abs_rad, norm_target_rad - np.pi)
        else:
            norm_target_rad = target_rad

        # computes the distance clockwise
        distance_rad = divmod(norm_target_rad - current_abs_rad, TWOPI)[1]

        # take the counter-clockwise distance if shorter
        if distance_rad > np.pi:
            distance_rad = distance_rad - TWOPI

        # if there are wrap limits
        if (min_abs_rad is not None) and (max_abs_rad is not None):
            # compute accumulated angle
            accum_abs_rad = current_abs_rad + distance_rad

            # if limits reached chose the other direction
            if accum_abs_rad > max_abs_rad - wrap_padding:
                distance_rad = distance_rad - TWOPI
            if accum_abs_rad < min_abs_rad + wrap_padding:
                distance_rad = distance_rad + TWOPI

        # compute final accumulated angle
        final_abs_rad = current_abs_rad + distance_rad

        return (final_abs_rad, distance_rad)

    #  Put in wrap padding kwarg so it's not used on camera rotation.
    def get_closest_state(self, targetposition, istracking=False):
        """Find the closest observatory state for the given target position.

        Parameters
        ----------
        targetposition : :class:`.ObservatoryPosition`
            A target position instance.
        istracking : bool, optional
            Flag for saying if the observatory is tracking. Default is False.

        Returns
        -------
        :class:`.ObservatoryState`
            The state that is closest to the current observatory state.

        Binary schema
        -------------
        The binary schema used to determine the state of a proposed target. A
        value of 1 indicates that is it failing. A value of 0 indicates that the
        state is passing.
        ___  ___  ___  ___  ___  ___
         |    |    |    |    |    |
        rot  rot  az   az   alt  alt
        max  min  max  min  max  min

        For example, if a proposed target exceeds the rotators maximum value,
        and is below the minimum azimuth we would have a binary value of;

         0    1    0    1    0    0

        If the target passed, then no limitations would occur;

         0    0    0    0    0    0
        """
        TWOPI = 2 * np.pi

        valid_state = True
        fail_record = self.current_state.fail_record
        self.current_state.fail_state = 0

        if targetposition.alt_rad < self.params.telalt_minpos_rad:
            telalt_rad = self.params.telalt_minpos_rad
            domalt_rad = self.params.telalt_minpos_rad
            valid_state = False

            if "telalt_minpos_rad" in fail_record:
                fail_record["telalt_minpos_rad"] += 1
            else:
                fail_record["telalt_minpos_rad"] = 1

            self.current_state.fail_state = self.current_state.fail_state | \
                                            self.current_state.fail_value_table["altEmin"]

        elif targetposition.alt_rad > self.params.telalt_maxpos_rad:
            telalt_rad = self.params.telalt_maxpos_rad
            domalt_rad = self.params.telalt_maxpos_rad
            valid_state = False
            if "telalt_maxpos_rad" in fail_record:
                fail_record["telalt_maxpos_rad"] += 1
            else:
                fail_record["telalt_maxpos_rad"] = 1

            self.current_state.fail_state = self.current_state.fail_state | \
                                            self.current_state.fail_value_table["altEmax"]

        else:
            telalt_rad = targetposition.alt_rad
            domalt_rad = targetposition.alt_rad

        if istracking:
            (telaz_rad, delta) = self.get_closest_angle_distance(targetposition.az_rad,
                                                                 self.current_state.telaz_rad)
            if telaz_rad < self.params.telaz_minpos_rad:
                telaz_rad = self.params.telaz_minpos_rad
                valid_state = False
                if "telaz_minpos_rad" in fail_record:
                    fail_record["telaz_minpos_rad"] += 1
                else:
                    fail_record["telaz_minpos_rad"] = 1

                self.current_state.fail_state = self.current_state.fail_state | \
                                                self.current_state.fail_value_table["azEmin"]

            elif telaz_rad > self.params.telaz_maxpos_rad:
                telaz_rad = self.params.telaz_maxpos_rad
                valid_state = False
                if "telaz_maxpos_rad" in fail_record:
                    fail_record["telaz_maxpos_rad"] += 1
                else:
                    fail_record["telaz_maxpos_rad"] = 1

                self.current_state.fail_state = self.current_state.fail_state | \
                                                self.current_state.fail_value_table["azEmax"]

        else:
            (telaz_rad, delta) = self.get_closest_angle_distance(targetposition.az_rad,
                                                                 self.current_state.telaz_rad,
                                                                 self.params.telaz_minpos_rad,
                                                                 self.params.telaz_maxpos_rad)

        (domaz_rad, delta) = self.get_closest_angle_distance(targetposition.az_rad,
                                                             self.current_state.domaz_rad)

        if istracking:
            (telrot_rad, delta) = self.get_closest_angle_distance(targetposition.rot_rad,
                                                                  self.current_state.telrot_rad,
                                                                  wrap_padding=0.)
            if telrot_rad < self.params.telrot_minpos_rad:
                telrot_rad = self.params.telrot_minpos_rad
                valid_state = False
                if "telrot_minpos_rad" in fail_record:
                    fail_record["telrot_minpos_rad"] += 1
                else:
                    fail_record["telrot_minpos_rad"] = 1

                self.current_state.fail_state = self.current_state.fail_state | \
                                                self.current_state.fail_value_table["rotEmin"]

            elif telrot_rad > self.params.telrot_maxpos_rad:
                telrot_rad = self.params.telrot_maxpos_rad
                valid_state = False
                if "telrot_maxpos_rad" in fail_record:
                    fail_record["telrot_maxpos_rad"] += 1
                else:
                    fail_record["telrot_maxpos_rad"] = 1

                self.current_state.fail_state = self.current_state.fail_state | \
                                                self.current_state.fail_value_table["rotEmax"]
        else:
            # if the target rotator angle is unreachable
            # then sets an arbitrary value (opposite)
            norm_rot_rad = divmod(targetposition.rot_rad - self.params.telrot_minpos_rad, TWOPI)[1] \
                + self.params.telrot_minpos_rad
            if norm_rot_rad > self.params.telrot_maxpos_rad:
                targetposition.rot_rad = norm_rot_rad - np.pi
            (telrot_rad, delta) = self.get_closest_angle_distance(targetposition.rot_rad,
                                                                  self.current_state.telrot_rad,
                                                                  self.params.telrot_minpos_rad,
                                                                  self.params.telrot_maxpos_rad,
                                                                  wrap_padding=0.)
        targetposition.ang_rad = divmod(targetposition.pa_rad - telrot_rad, TWOPI)[1]

        targetstate = ObservatoryState()
        targetstate.set_position(targetposition)
        targetstate.telalt_rad = telalt_rad
        targetstate.telaz_rad = telaz_rad
        targetstate.telrot_rad = telrot_rad
        targetstate.domalt_rad = domalt_rad
        targetstate.domaz_rad = domaz_rad
        if istracking:
            targetstate.tracking = valid_state

        self.current_state.fail_record = fail_record

        return targetstate


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


class Model_observatory(object):
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

        self.scheduledDowntime_model = sdt
        self.unscheduledDowntime_model = usdt

        for downtime in sdt.downtimes:
            self.down_nights.extend(range(downtime[0], downtime[0]+downtime[1], 1))
        for downtime in usdt.downtimes:
            self.down_nights.extend(range(downtime[0], downtime[0]+downtime[1], 1))
        self.down_nights.sort()

        self.seeing_model = SeeingSim(dth)
        self.seeing_indx_dict = {}
        for i, filtername in enumerate(self.seeing_model.filter_list):
            self.seeing_indx_dict[filtername] = i

        self.cloud_model = CloudModel(dth)
        self.cloud_model.read_data()
        # XXX-argle bargle, really?!?!?
        self.cloud_model.offset = 0

        self.sky_model = sb.SkyModelPre(speedLoad=quickTest)

        self.observatory = ExtendedObservatoryModel()
        self.observatory.configure_from_module()
        # Make it so it respects my requested rotator angles
        self.observatory.params.rotator_followsky = True

        self.filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
        self.seeing_FWHMeff = {}
        for key in self.filterlist:
            self.seeing_FWHMeff[key] = np.zeros(hp.nside2npix(self.nside), dtype=float)

        self.almanac = Almanac(mjd_start=mjd_start)

        # Let's make sure we're at an openable MJD
        good_mjd = False
        to_set_mjd = mjd_start
        while not good_mjd:
            good_mjd, to_set_mjd = self.check_mjd(to_set_mjd)
        self.mjd = to_set_mjd

        self.obsID_counter = 0

    def get_info(self):
        """
        Returns
        -------
        Array with model versions that were instantiated
        """

        # The things we want to get info on
        models = {'cloud model': self.cloud_model, 'sky model': self.sky_model,
                  'seeing model': self.seeing_model, 'observatory model': self.observatory,
                  'sched downtime model': self.scheduledDowntime_model,
                  'unched downtime model': self.unscheduledDowntime_model}

        result = []
        for model_name in models:
            try:
                module_name = models[model_name].__module__
                module = import_module(module_name)
                ver = import_module(module.__package__+'.version')
                version = ver.__version__
                fingerprint = ver.__fingerprint__
            except:
                version = 'NA'
                fingerprint = 'NA'
            result.append([model_name+' version', version])
            result.append([model_name+' fingerprint', fingerprint])
            result.append([model_name+' module', models[model_name].__module__])

        return result

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

        # Clouds. Add a +0 because this might alter input!!!
        self.conditions.bulk_cloud = self.cloud_model.get_cloud(delta_t+0)

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
        self.conditions.current_filter = self.observatory.current_state.filter[0]

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
        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        # convert these to scalars
        for key in sun_moon_info:
            sun_moon_info[key] = sun_moon_info[key].max()
        self.conditions.moonPhase = sun_moon_info['moon_phase']

        self.conditions.moonAlt = sun_moon_info['moon_alt']
        self.conditions.moonAz = sun_moon_info['moon_az']
        self.conditions.moonRA = sun_moon_info['moon_RA']
        self.conditions.moonDec = sun_moon_info['moon_dec']
        self.conditions.sunAlt = sun_moon_info['sun_alt']
        self.conditions.sunRA = sun_moon_info['sun_RA']
        self.conditions.sunDec = sun_moon_info['sun_dec']

        self.conditions.lmst, last = calcLmstLast(self.mjd, self.site.longitude_rad)

        self.conditions.telRA = self.observatory.current_state.ra_rad
        self.conditions.telDec = self.observatory.current_state.dec_rad

        self.conditions.rotTelPos = self.observatory.current_state.rot_rad

        # Add in the almanac information
        self.conditions.night = self.night
        self.conditions.sunset = self.almanac.sunsets['sunset'][self.almanac_indx]
        self.conditions.sun_n12_setting = self.almanac.sunsets['sun_n12_setting'][self.almanac_indx]
        self.conditions.sun_n18_setting = self.almanac.sunsets['sun_n18_setting'][self.almanac_indx]
        self.conditions.sun_n18_rising = self.almanac.sunsets['sun_n18_rising'][self.almanac_indx]
        self.conditions.sun_n12_rising = self.almanac.sunsets['sun_n12_rising'][self.almanac_indx]
        self.conditions.sunrise = self.almanac.sunsets['sunrise'][self.almanac_indx]
        self.conditions.moonrise = self.almanac.sunsets['moonrise'][self.almanac_indx]
        self.conditions.moonset = self.almanac.sunsets['moonset'][self.almanac_indx]

        return self.conditions

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = self.almanac.mjd_indx(value)
        self.night = self.almanac.sunsets['night'][self.almanac_indx]

    def observation_add_data(self, observation):
        """
        Fill in the metadata for a completed observation
        """
        # Time since start of simulation
        delta_t = (self.mjd-self.mjd_start)*24.*3600.

        observation['clouds'] = self.cloud_model.get_cloud(delta_t)
        observation['airmass'] = 1./np.cos(np.pi/2. - observation['alt'])
        # Seeing
        fwhm_500, fwhm_eff, fwhm_geom = self.seeing_model.get_seeing(delta_t, observation['airmass'])
        observation['FWHMeff'] = fwhm_eff[self.seeing_indx_dict[observation['filter'][0]]]
        observation['FWHM_geometric'] = fwhm_geom[self.seeing_indx_dict[observation['filter'][0]]]
        observation['FWHM_500'] = fwhm_500

        observation['night'] = self.night
        observation['mjd'] = self.mjd

        hpid = _raDec2Hpid(self.sky_model.nside, observation['RA'], observation['dec'])
        observation['skybrightness'] = self.sky_model.returnMags(self.mjd,
                                                                 indx=[hpid],
                                                                 extrapolate=True)[observation['filter'][0]]

        observation['fivesigmadepth'] = m5_flat_sed(observation['filter'][0], observation['skybrightness'],
                                                    observation['FWHMeff'], observation['exptime'],
                                                    observation['airmass'])

        lmst, last = calcLmstLast(self.mjd, self.site.longitude_rad)
        observation['lmst'] = lmst

        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        observation['sunAlt'] = sun_moon_info['sun_alt']
        observation['sunAz'] = sun_moon_info['sun_az']
        observation['sunRA'] = sun_moon_info['sun_RA']
        observation['sunDec'] = sun_moon_info['sun_dec']
        observation['moonAlt'] = sun_moon_info['moon_alt']
        observation['moonAz'] = sun_moon_info['moon_az']
        observation['moonRA'] = sun_moon_info['moon_RA']
        observation['moonDec'] = sun_moon_info['moon_dec']
        observation['moonDist'] = _angularSeparation(observation['RA'], observation['dec'],
                                                     observation['moonRA'], observation['moonDec'])
        observation['solarElong'] = _angularSeparation(observation['RA'], observation['dec'],
                                                       observation['sunRA'], observation['sunDec'])
        observation['moonPhase'] = sun_moon_info['moon_phase']

        observation['ID'] = self.obsID_counter
        self.obsID_counter += 1

        return observation

    def check_mjd(self, mjd, cloud_skip=20.):
        """See if an mjd is ok to observe
        Parameters
        ----------
        cloud_skip : float (20)
            How much time to skip ahead if it's cloudy (minutes)


        Returns
        -------
        bool

        mdj : float
            If True, the input mjd. If false, a good mjd to skip forward to.
        """
        passed = True
        new_mjd = mjd + 0
        delta_t = (mjd-self.mjd_start)*24.*3600.
        clouds = self.cloud_model.get_cloud(delta_t+0.)
        # Maybe set this to a while loop to make sure we don't land on another cloudy time?
        # or just make this an entire recursive call?
        if clouds > self.cloud_limit:
            passed = False
            new_mjd = mjd + cloud_skip/60./24.
        alm_indx = np.searchsorted(self.almanac.sunsets['sunset'], mjd) - 1
        # at the end of the night, advance to the next setting twilight
        if mjd > self.almanac.sunsets['sun_n12_rising'][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        if mjd < self.almanac.sunsets['sun_n12_setting'][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        # We're in a down night, advance to next night
        if self.almanac.sunsets['night'][alm_indx] in self.down_nights:
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        # recursive call to make sure we skip far enough ahead
        if not passed:
            while not passed:
                passed, new_mjd = self.check_mjd(new_mjd)
            return False, new_mjd
        else:
            return True, mjd

    def observe(self, observation):
        """Try to make an observation

        Returns
        -------
        observation : observation object
            None if there was no observation taken. Completed observation with meta data filled in.
        new_night : bool
            Have we started a new night.
        """

        start_night = self.night.copy()

        # Make sure the kinematic model is set to the correct mjd
        t = Time(self.mjd, format='mjd')
        self.observatory.update_state(t.unix)

        target = Target(band_filter=observation['filter'], ra_rad=observation['RA'],
                        dec_rad=observation['dec'], ang_rad=observation['rotSkyPos'],
                        num_exp=observation['nexp'], exp_times=[observation['exptime']])
        start_ra = self.observatory.current_state.ra_rad
        start_dec = self.observatory.current_state.dec_rad
        slewtime, visittime = self.observatory.observe_times(target)

        # Check if the mjd after slewtime and visitime is fine:
        observation_worked, new_mjd = self.check_mjd(self.mjd + (slewtime + visittime)/24./3600.)
        if observation_worked:
            observation['visittime'] = visittime
            observation['slewtime'] = slewtime
            observation['slewdist'] = _angularSeparation(start_ra, start_dec,
                                                         self.observatory.current_state.ra_rad,
                                                         self.observatory.current_state.dec_rad)
            self.mjd = self.mjd + slewtime/24./3600.
            # Reach into the observatory model to pull out the relevant data it has calculated
            # Note, this might be after the observation has been completed.
            observation['alt'] = self.observatory.current_state.alt_rad
            observation['az'] = self.observatory.current_state.az_rad
            observation['pa'] = self.observatory.current_state.pa_rad
            observation['rotTelPos'] = self.observatory.current_state.rot_rad
            observation['rotSkyPos'] = self.observatory.current_state.ang_rad

            # Metadata on observation is after slew and settle, so at start of exposure.
            result = self.observation_add_data(observation)
            self.mjd = self.mjd + visittime/24./3600.
            new_night = False
        else:
            result = None
            self.observatory.park()
            # Skip to next legitimate mjd
            self.mjd = new_mjd
            now_night = self.night
            if now_night == start_night:
                new_night = False
            else:
                new_night = True

        return result, new_night
