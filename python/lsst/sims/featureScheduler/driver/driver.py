
import numpy as np
from importlib import import_module
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
from lsst.ts.scheduler.proposals import AreaDistributionProposal
import logging

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

        self.scheduler = None
        self.sky_nside = 32
        self.scheduler_visit_counting_bfs = 0

        self.proposal_id_dict = {}


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

    def create_area_proposal(self, propid, name, config_dict):
        '''Override create_area_proposal from superclass.

        :param propid:
        :param name:
        :param config_dict:
        :return:
        '''
        # Load configuration from a python module.
        # TODO:
        # This can be changed later to load from a given string, I'm planning on getting the name from
        # lsst.sims.ocs.configuration.survey.general_proposals
        conf = import_module('lsst.sims.featureScheduler.driver.config')

        self.scheduler = conf.scheduler
        self.sky_nside = conf.nside
        self.scheduler_visit_counting_bfs = conf.scheduler_visit_counting_bfs

        # Now, configure the different proposals inside Driver
        # Get the proposal list from feature based scheduler
        # proposal_id_list = np.array([])
        # proposal_name_list = np.array([])
        for prop in self.scheduler.surveys:
            for i, pid in enumerate(prop.basis_functions[self.scheduler_visit_counting_bfs].id_list):
                self.proposal_id_dict[pid] = [0,
                                              prop.basis_functions[self.scheduler_visit_counting_bfs].name_list[i]]

            # np.append(proposal_id_list,
            #           prop.basis_functions[self.scheduler_visit_counting_bfs].id_list)
            # np.append(proposal_name_list,
            #           prop.basis_functions[self.scheduler_visit_counting_bfs].name_list)

        # proposal_id_list = np.unique(proposal_id_list)
        # proposal_name_list = np.unique(proposal_name_list)

        for pid in self.proposal_id_dict.keys():
            self.proposal_id_dict[pid][0] = self.propid_counter
            self.propid_counter += 1

            self.log.debug('%s: %s' % (pid, self.proposal_id_dict[pid]))

            area_prop = AreaDistributionProposal(pid,
                                                 self.proposal_id_dict[pid][1],
                                                 config_dict,
                                                 self.sky)
            area_prop.configure_constraints(self.params)
            self.science_proposal_list.append(area_prop)

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
                filter_visits_dict[filter] = \
                    np.sum(prop.basis_functions[self.scheduler_visit_counting_bfs].survey_features['N_obs'].feature)
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
        telemetry_stream['mounted_filters'] = self.observatoryModel.current_state.mountedfilters
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

        self.scheduler_winner_target = winner_target

        hpid = _raDec2Hpid(self.sky_nside, winner_target['RA'][0], winner_target['dec'][0])
        # Fixme: How to determine the survey that generated the target? Im assuming it was the first one
        # self.log.debug('target_hpid: %i ' % hpid)
        # self.log.debug('target: %s' % winner_target)

        propid = winner_target['survey_id'][0]
        filtername = winner_target['filter'][0]
        indx = self.proposal_id_dict[propid][0]

        if winner_target['field_id'][0] in self.target_list:
            if winner_target['filter'][0] in self.target_list[winner_target['field_id'][0]]:
                target = self.target_list[winner_target['field_id'][0]][winner_target['filter'][0]]
            else:
                target = self.generate_target(winner_target[0])
                self.target_list[winner_target['field_id'][0]][winner_target['filter'][0]] = target
                self.science_proposal_list[indx].survey_targets_dict[target.fieldid][filtername] = target
        else:
            target = self.generate_target(winner_target[0])
            self.target_list[target.fieldid] = {filtername: target}
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {filtername: target}
        target.time = self.time
        # target.propid = [1]

        slewtime = self.observatoryModel.get_slew_delay(target)
        if slewtime > 0.:
            self.scheduler_winner_target['mjd'] = telemetry_stream['mjd']+slewtime/60./60./24.
            self.scheduler_winner_target['night'] = self.night
            self.scheduler_winner_target['slewtime'] = slewtime
            self.scheduler_winner_target['skybrightness'] = \
                self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd,
                                               indx=[hpid],
                                               extrapolate=True)[filtername]
            self.scheduler_winner_target['FWHMeff'] = telemetry_stream['FWHMeff_%s' % filtername][hpid]
            self.scheduler_winner_target['FWHM_geometric'] = \
                telemetry_stream['FWHM_geometric_%s' % winner_target['filter'][0]][hpid]
            self.scheduler_winner_target['airmass'] = telemetry_stream['airmass'][hpid]
            self.scheduler_winner_target['fivesigmadepth'] = m5_flat_sed(filtername,
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
            target.seeing = self.seeing
            target.airmass = telemetry_stream['airmass'][hpid]
            target.sky_brightness = self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd,
                                                                   indx=[hpid],
                                                                   extrapolate=True)[filtername][0]

            self.log.debug(target)

            self.last_winner_target = target.get_copy()
        else:
            self.last_winner_target = self.nulltarget

        return self.last_winner_target

    def register_observation(self, observation):
        # obs_hpid = np.where(self.scheduler.surveys[0].hp2fields == self.scheduler_winner_target['survey_id'])
        self.scheduler.add_observation(self.scheduler_winner_target)
        # self.log.debug('propid: %i' % self.last_winner_target.propid)
        # self.log.debug(self.proposal_id_dict)
        idx = self.proposal_id_dict[self.last_winner_target.propid][0]
        self.science_proposal_list[idx].winners_list = [observation]

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

    def generate_target(self, fb_observation):
        '''Takes an observation array given by the feature based scheduler and generate an appropriate OpSim target.

        :param fb_observation: numpy.array
        :return: Target
        '''

        self.targetid += 1
        filtername = fb_observation['filter']
        propid = fb_observation['survey_id']

        target = Target()
        target.targetid = self.targetid
        target.fieldid = fb_observation['field_id']
        target.filter = str(filtername)
        target.num_exp = fb_observation['nexp']
        target.exp_times = [fb_observation['exptime'] / fb_observation['nexp']] * fb_observation['nexp']
        target.ra_rad = fb_observation['RA']
        target.dec_rad = fb_observation['dec']
        target.propid = propid
        target.goal = 100
        target.visits = 0
        target.progress = 0.0
        target.groupid = -1
        target.groupix = -1
        target.propid_list = [propid]
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

        return target
