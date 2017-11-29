
import numpy as np
from numpy.lib.recfunctions import append_fields

from lsst.sims.ocs.environment import SeeingModel, CloudModel
from lsst.sims.ocs.configuration import Environment
from lsst.sims.ocs.configuration.instrument import Filters
from lsst.sims.ocs.kernel.time_handler import TimeHandler
from lsst.sims.skybrightness_pre import SkyModelPre
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, calcLmstLast, m5_flat_sed

from lsst.ts.observatory.model import Target
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler import stupidFast_RaDec2AltAz
from lsst.ts.dateloc import DateProfile
from lsst.ts.scheduler import Driver
import healpy as hp
import logging

# from model_notime import SeeingModel_no_time, CloudModel_no_time

__all__ = ["FeatureSchedulerDriver"]


class FeatureSchedulerDriver(Driver):

    def __init__(self):

        Driver.__init__(self)
        self.log = logging.getLogger("featureSchedulerDriver")

        # self.obsloc = ObservatoryLocation()
        # self.obsloc.for_lsst()
        # self.dateprofile = DateProfile(0, self.obsloc)

        # FIXME: This should probably come from outside telemetry stream. But right now, cloud and seeing are only
        # floats. Needs to be fixed externally.
        self.scheduler_time_handle = TimeHandler("2022-10-01")  # FIXME: Hard coded config! ideally this won't be here
        self.cloud_model = CloudModel(self.scheduler_time_handle)
        self.seeing_model = SeeingModel(self.scheduler_time_handle)
        self.cloud_model.initialize()
        self.seeing_model.initialize(Environment(), Filters())
        self.sky_brightness = SkyModelPre()  # The sky brightness in self.sky uses opsim fields. We need healpix here

        self.night = 0

        self.target_list = {}

        # Fixme: hardcoded configuration
        target_maps = {}
        nside = fs.set_default_nside(nside=32)
        target_maps['u'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.31, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)
        target_maps['g'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.44, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)
        target_maps['r'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=1.0, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)
        target_maps['i'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=1.0, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)
        target_maps['z'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.9, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)
        target_maps['y'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.9, SCP_fraction=0.,
                                        GP_fraction=0., nside=nside)

        filters = self.observatoryModel.filters
        surveys = []

        for filtername in filters:
            bfs = []
            bfs.append(fs.M5_diff_basis_function(filtername=filtername, nside=nside))
            bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                    target_map=target_maps[filtername],
                                                    out_of_bounds_val=hp.UNSEEN, nside=nside))

            bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50., nside=nside))
            # bfs.append(fs.Zenith_mask_basis_function(maxAlt=78., penalty=-100, nside=nside))
            bfs.append(fs.Slewtime_basis_function(filtername=filtername, nside=nside))
            bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

            weights = np.array([3.0, 0.2, 1., 3., 3.])
            surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=False,
                                                   nside=nside,
                                                   prop_id=1,
                                                   prop_name='WFD'))

        self.scheduler = fs.Core_scheduler(surveys, nside=nside)
        self.sky_nside = nside

    def start_survey(self, timestamp, night):

        self.start_time = timestamp
        self.log.info("start_survey t=%.6f" % timestamp)

        self.survey_started = True

        self.sky.update(timestamp)

        (sunset, sunrise) = self.sky.get_night_boundaries(self.params.night_boundary)
        self.log.debug("start_survey sunset=%.6f sunrise=%.6f" % (sunset, sunrise))
        if sunset <= timestamp < sunrise:
            self.start_night(timestamp, night)

        self.sunset_timestamp = sunset
        self.sunrise_timestamp = sunrise

    def end_survey(self):

        self.log.info("end_survey")

    def start_night(self, timestamp, night):

        timeprogress = (timestamp - self.start_time) / self.survey_duration_SECS
        self.log.info("start_night t=%.6f, night=%d timeprogress=%.2f%%" %
                      (timestamp, night, 100 * timeprogress))

        self.isnight = True
        self.night = night

    def end_night(self, timestamp, night):

        timeprogress = (timestamp - self.start_time) / self.survey_duration_SECS
        self.log.info("end_night t=%.6f, night=%d timeprogress=%.2f%%" %
                      (timestamp, night, 100 * timeprogress))

        self.isnight = False

        self.last_winner_target = self.nulltarget
        self.deep_drilling_target = None

        total_filter_visits_dict = {}

        for prop in self.scheduler.surveys:
            filter_visits_dict = {}
            for filter in self.observatoryModel.filters:
                if filter not in total_filter_visits_dict:
                    total_filter_visits_dict[filter] = 0
                filter_visits_dict[filter] = np.sum(prop.basis_functions[1].survey_features['N_obs'].feature)
                total_filter_visits_dict[filter] += filter_visits_dict[filter]
                self.log.debug("end_night propid=%d name=%s filter=%s visits=%i" %
                               (prop.prop_id, prop.prop_name, filter, filter_visits_dict[filter]))

        previous_midnight_moonphase = self.midnight_moonphase
        self.sky.update(timestamp)
        (sunset, sunrise) = self.sky.get_night_boundaries(self.params.night_boundary)
        self.log.debug("end_night sunset=%.6f sunrise=%.6f" % (sunset, sunrise))

        self.sunset_timestamp = sunset
        self.sunrise_timestamp = sunrise
        next_midnight = (sunset + sunrise) / 2
        self.sky.update(next_midnight)
        info = self.sky.get_moon_sun_info(np.array([0.0]), np.array([0.0]))
        self.midnight_moonphase = info["moonPhase"]
        self.log.info("end_night next moonphase=%.2f%%" % (self.midnight_moonphase))

        self.need_filter_swap = False
        self.filter_to_mount = ""
        self.filter_to_unmount = ""
        if self.darktime:
            if self.midnight_moonphase > previous_midnight_moonphase:
                self.log.info("end_night dark time waxing")
                if self.midnight_moonphase > self.params.new_moon_phase_threshold:
                    self.need_filter_swap = True
                    self.filter_to_mount = self.unmounted_filter
                    self.filter_to_unmount = self.mounted_filter
                    self.darktime = False
            else:
                self.log.info("end_night dark time waning")
        else:
            if self.midnight_moonphase < previous_midnight_moonphase:
                self.log.info("end_night bright time waning")
                if self.midnight_moonphase < self.params.new_moon_phase_threshold:
                    self.need_filter_swap = True
                    self.filter_to_mount = self.observatoryModel.params.filter_darktime
                    max_progress = -1.0
                    for filter in self.observatoryModel.params.filter_removable_list:
                        if total_filter_visits_dict[filter] > max_progress:
                            self.filter_to_unmount = filter
                            max_progress = total_filter_visits_dict[filter]
                    self.darktime = True
            else:
                self.log.info("end_night bright time waxing")

        if self.need_filter_swap:
            self.log.debug("end_night filter swap %s=>cam=>%s" %
                           (self.filter_to_mount, self.filter_to_unmount))

    def select_next_target(self):

        if not self.isnight:
            return self.nulltarget

        # Telemetry stream
        telemetry_stream = {}

        telemetry_stream['mjd'] = self.observatoryModel.dateprofile.mjd
        telemetry_stream['night'] = self.night
        telemetry_stream['lmst'] = self.observatoryModel.dateprofile.lst_rad*12./np.pi

        dp = DateProfile(0, self.observatoryModel.dateprofile.location)
        mjd, _ = dp(self.sunrise_timestamp)
        telemetry_stream['next_twilight_start'] = dp.mjd

        dp = DateProfile(0, self.observatoryModel.dateprofile.location)
        mjd, _ = dp(self.sunset_timestamp)
        telemetry_stream['next_twilight_end'] = dp.mjd
        telemetry_stream['last_twilight_end'] = dp.mjd

        # Telemetry about where the observatory is pointing and what filter it's using.
        telemetry_stream['filter'] = self.observatoryModel.current_state.filter
        telemetry_stream['telRA'] = np.degrees(self.observatoryModel.current_state.ra_rad)
        telemetry_stream['telDec'] = np.degrees(self.observatoryModel.current_state.dec_rad)
        telemetry_stream['telAlt'] = np.degrees(self.observatoryModel.current_state.alt_rad)
        telemetry_stream['telAz'] = np.degrees(self.observatoryModel.current_state.az_rad)
        # telemetry_stream['telRA'] = self.observatoryModel.current_state.ra_rad
        # telemetry_stream['telDec'] = self.observatoryModel.current_state.dec_rad
        # telemetry_stream['telAlt'] = self.observatoryModel.current_state.alt_rad
        # telemetry_stream['telAz'] = self.observatoryModel.current_state.az_rad

        # What is the sky brightness over the sky (healpix map)
        telemetry_stream['skybrightness'] = self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd)

        # Find expected slewtimes over the sky.
        # The slewtimes telemetry is expected to be a healpix map of slewtimes, in the current filter.
        # There are other features that balance the filter change cost, separately.
        # (this filter aspect may be something to revisit in the future?)
        # Note that these slewtimes are -1 where the telescope is not allowed to point.
        alt, az = stupidFast_RaDec2AltAz(self.scheduler.ra_grid_rad,
                                         self.scheduler.dec_grid_rad,
                                         self.observatoryModel.location.latitude_rad,
                                         self.observatoryModel.location.longitude_rad,
                                         self.observatoryModel.dateprofile.mjd,
                                         self.observatoryModel.dateprofile.lst_rad)
        current_filter = self.observatoryModel.current_state.filter

        telemetry_stream['slewtimes'] = self.observatoryModel.get_approximate_slew_delay(alt, az,
                                                                                         current_filter)
        # What is the airmass over the sky (healpix map).

        telemetry_stream['airmass'] = self.sky_brightness.returnAirmass(self.observatoryModel.dateprofile.mjd)
        delta_t = (self.time-self.start_time)
        telemetry_stream['clouds'] = self.cloud_model.get_cloud(int(delta_t))
        for filtername in ['u', 'g', 'r', 'i', 'z', 'y']:
            fwhm_500, fwhm_geometric, fwhm_effective = self.seeing_model.calculate_seeing(delta_t, filtername,
                                                                                          telemetry_stream['airmass'])
            telemetry_stream['FWHMeff_%s' % filtername] = fwhm_effective  # arcsec
            telemetry_stream['FWHM_geometric_%s' % filtername] = fwhm_geometric

        sunMoon_info = self.sky_brightness.returnSunMoon(self.observatoryModel.dateprofile.mjd)
        # Pretty sure these are radians
        telemetry_stream['sunAlt'] = np.max(sunMoon_info['sunAlt'])
        telemetry_stream['moonAlt'] = np.max(sunMoon_info['moonAlt'])

        dict_of_lists = {}
        for key in telemetry_stream:
            dict_of_lists[key] = np.array(telemetry_stream[key])

        self.scheduler.update_conditions(telemetry_stream)
        winner_target = self.scheduler.request_observation()

        self.log.debug(winner_target)
        self.scheduler_winner_target = winner_target

        target = Target()

        if winner_target['field_id'][0] in self.target_list:
            if winner_target['filter'][0] in self.target_list[winner_target['field_id'][0]]:
                target = self.target_list[winner_target['field_id'][0]][winner_target['filter'][0]]
            else:
                self.targetid += 1
                target.targetid = self.targetid
                target.fieldid = winner_target['field_id'][0]
                target.filter = str(winner_target['filter'][0])
                target.num_exp = winner_target['nexp'][0]
                target.exp_times = [winner_target['exptime'][0]/winner_target['nexp'][0]] * winner_target['nexp'][0]
                target.ra_rad = winner_target['RA'][0]
                target.dec_rad = winner_target['dec'][0]
                target.propid = 1
                target.goal = 100
                target.visits = 0
                target.progress = 0.0
                target.groupid = -1
                target.groupix = -1
                target.propid_list = [1]
                target.need_list = [target.need]
                target.bonus_list = [target.bonus]
                target.value_list = [target.value]
                target.propboost_list = [target.propboost]
                target.sequenceid_list = [target.sequenceid]
                target.subsequencename_list = [target.subsequencename]
                target.groupid_list = [target.groupid]
                target.groupix_list = [target.groupix]
                target.is_deep_drilling_list = [target.is_deep_drilling]
                target.is_dd_firstvisit_list = [target.is_dd_firstvisit]
                target.remaining_dd_visits_list = [target.remaining_dd_visits]
                target.dd_exposures_list = [target.dd_exposures]
                target.dd_filterchanges_list = [target.dd_filterchanges]
                target.dd_exptime_list = [target.dd_exptime]

                self.target_list[winner_target['field_id'][0]][winner_target['filter'][0]] = target
                self.science_proposal_list[0].survey_targets_dict[target.fieldid][winner_target['filter'][0]] = \
                    target
        else:
            self.targetid += 1
            target.targetid = self.targetid
            target.fieldid = winner_target['field_id'][0]
            target.filter = str(winner_target['filter'][0])
            target.num_exp = winner_target['nexp'][0]
            target.exp_times = [winner_target['exptime'][0]/winner_target['nexp'][0]] * winner_target['nexp'][0]
            target.ra_rad = winner_target['RA'][0]
            target.dec_rad = winner_target['dec'][0]
            target.propid = 1
            target.goal = 100
            target.visits = 0
            target.progress = 0.0
            target.groupid = -1
            target.groupix = -1
            target.propid_list = [1]
            target.need_list = [target.need]
            target.bonus_list = [target.bonus]
            target.value_list = [target.value]
            target.propboost_list = [target.propboost]
            target.sequenceid_list = [target.sequenceid]
            target.subsequencename_list = [target.subsequencename]
            target.groupid_list = [target.groupid]
            target.groupix_list = [target.groupix]
            target.is_deep_drilling_list = [target.is_deep_drilling]
            target.is_dd_firstvisit_list = [target.is_dd_firstvisit]
            target.remaining_dd_visits_list = [target.remaining_dd_visits]
            target.dd_exposures_list = [target.dd_exposures]
            target.dd_filterchanges_list = [target.dd_filterchanges]
            target.dd_exptime_list = [target.dd_exptime]
            self.target_list[winner_target['field_id'][0]] = {winner_target['filter'][0]: target}
            self.science_proposal_list[0].survey_targets_dict[target.fieldid] = {winner_target['filter'][0]: target}
        target.time = self.time
        # target.propid = [1]

        slewtime = self.observatoryModel.get_slew_delay(target)
        if slewtime > 0.:
            hpid = _raDec2Hpid(self.sky_nside, target.ra_rad, target.dec_rad)
            self.scheduler_winner_target['mjd'] = telemetry_stream['mjd']+slewtime/60./60./24.
            self.scheduler_winner_target['night'] = self.night
            self.scheduler_winner_target['slewtime'] = slewtime
            self.scheduler_winner_target['skybrightness'] = \
                self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd,
                                               indx=[hpid],
                                               extrapolate=True)[winner_target['filter'][0]]
            self.scheduler_winner_target['FWHMeff'] = telemetry_stream['FWHMeff_%s' % winner_target['filter'][0]][hpid]
            self.scheduler_winner_target['FWHM_geometric'] = \
                telemetry_stream['FWHM_geometric_%s' % winner_target['filter'][0]][hpid]
            self.scheduler_winner_target['airmass'] = telemetry_stream['airmass'][hpid]
            self.scheduler_winner_target['fivesigmadepth'] = m5_flat_sed(self.scheduler_winner_target['filter'][0],
                                                        self.scheduler_winner_target['skybrightness'],
                                                        self.scheduler_winner_target['FWHMeff'],
                                                        self.scheduler_winner_target['exptime'],
                                                        self.scheduler_winner_target['airmass'])
            self.scheduler_winner_target['alt'] = target.alt_rad
            self.scheduler_winner_target['az'] = target.az_rad
            self.scheduler_winner_target['clouds'] = self.cloud
            self.scheduler_winner_target['sunAlt'] = telemetry_stream['sunAlt']
            self.scheduler_winner_target['moonAlt'] = telemetry_stream['moonAlt']

            self.observatoryModel2.set_state(self.observatoryState)
            self.observatoryModel2.observe(target)

        # self.log.debug(target)

            self.last_winner_target = target.get_copy()
        else:
            self.last_winner_target = self.nulltarget

        self.log.debug(self.scheduler_winner_target)
        return self.last_winner_target

    def register_observation(self, observation):

        self.log.debug('Registering: %s' % self.scheduler_winner_target)
        self.scheduler.add_observation(self.scheduler_winner_target)
        self.science_proposal_list[0].winners_list = [observation]

        return super(FeatureSchedulerDriver, self).register_observation(observation)

    def update_time(self, timestamp, night):

        delta_time = timestamp-self.time
        self.time = timestamp
        self.scheduler_time_handle.update_time(delta_time, 'seconds')
        self.observatoryModel.update_state(self.time)

        if not self.survey_started:
            self.start_survey(timestamp, night)

        if self.isnight:
            # if round(timestamp) >= round(self.sunrise_timestamp):
            if timestamp >= self.sunrise_timestamp:
                self.end_night(timestamp, night)
        else:
            # if round(timestamp) >= round(self.sunset_timestamp):
            if timestamp >= self.sunset_timestamp:
                self.start_night(timestamp, night)

        return self.isnight
