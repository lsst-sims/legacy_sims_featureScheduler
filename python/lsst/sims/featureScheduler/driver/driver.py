
import os
import copy
import numpy as np
from importlib import import_module
import importlib.util

from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, calcLmstLast, m5_flat_sed, _approx_RaDec2AltAz
from lsst.sims.seeingModel import SeeingModel

from lsst.sims.featureScheduler.driver.constants import CONFIG_NAME

from lsst.ts.observatory.model import Target
from lsst.sims.featureScheduler import obs_to_fbsobs
from lsst.ts.dateloc import DateProfile
from lsst.ts.scheduler import Driver

import logging

__all__ = ["FeatureSchedulerDriver"]


class FeatureSchedulerDriver(Driver):

    def __init__(self):

        Driver.__init__(self)
        self.log = logging.getLogger("featureSchedulerDriver")

        # FIXME: Get parameters for the seeing model!
        telescope_seeing = 0.25
        optical_design_seeing = 0.08
        camera_seeing = 0.3

        self.seeing_model = SeeingModel(telescope_seeing=telescope_seeing,
                                        optical_design_seeing=optical_design_seeing,
                                        camera_seeing=camera_seeing)

        # self.sky_brightness = SkyModelPre()  # The sky brightness in self.sky uses opsim fields. We need healpix here
        self.sky_brightness = self.sky.sky_brightness
        self.night = 0

        self.target_list = {}

        self.scheduler = None
        self.sky_nside = 32
        self.scheduler_visit_counting_bfs = 0

        self.proposal_id_dict = {}

        self.time_distribution = False

        self.initialized = False

    # def start_survey(self, timestamp, night):
    #
    #     self.start_time = timestamp
    #     self.log.info("start_survey t=%.6f" % timestamp)
    #
    #     self.survey_started = True
    #
    #     self.sky.update(timestamp)
    #
    #     (sunset, sunrise) = self.sky.get_night_boundaries(self.params.night_boundary)
    #     self.log.debug("start_survey sunset=%.6f sunrise=%.6f" % (sunset, sunrise))
    #     if sunset <= timestamp < sunrise:
    #         self.start_night(timestamp, night)
    #
    #     self.sunset_timestamp = sunset
    #     self.sunrise_timestamp = sunrise

    # def create_area_proposal(self, propid, name, config_dict):
    #     '''Override create_area_proposal from superclass.
    #
    #     One call to rule them all!
    #
    #     :param propid:
    #     :param name:
    #     :param config_dict:
    #     :return:
    #     '''
    #     if not self.initialized and name == 'WideFastDeep':
    #         self.initialize(config_dict)
    #
    # def create_sequence_proposal(self, propid, name, config_dict):
    #     pass

    def configure_scheduler(self, **kwargs):
        """
        Load configuration from a python module.

        :param kwargs:
        :return:
        """

        # Check if there's a feature scheduler configuration file on the path
        if 'config_name' in kwargs:
            configure_path = os.path.join(kwargs['config_path'], kwargs['config_name'])
        else:
            configure_path = os.path.join(kwargs['config_path'], CONFIG_NAME)

        force = kwargs.pop('force', False)
        survey_topology = None

        if self.scheduler is None or force:
            if os.path.exists(configure_path):
                self.log.info('Loading feature based scheduler configuration from {}.'.format(configure_path))
                spec = importlib.util.spec_from_file_location("config", configure_path)
                conf = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(conf)
                # conf = None
            else:
                self.log.info('Loading feature based scheduler default configuration.')
                conf = import_module('lsst.sims.featureScheduler.driver.config')

            self.scheduler = conf.scheduler
            self.sky_nside = conf.nside
            survey_topology = conf.survey_topology
        else:
            self.log.info('Scheduler already configured.')

        for survey_list in self.scheduler.survey_lists:
            for survey in survey_list:
                if 'proposals' in survey.features:
                    for i, pid in enumerate(survey.features['proposals'].id.keys()):
                        # This gets names of all proposals on all surveys, overwrites repeated and stores new ones
                        self.proposal_id_dict[pid] = [0,
                                                      survey.features['proposals'].id[pid]]

        self.log.debug('Proposal_id_dict: %s' % self.proposal_id_dict.keys())
        # self.log.debug('survey_fields: %s' % survey_fields.keys())

        for pid in self.proposal_id_dict.keys():
            self.proposal_id_dict[pid][0] = self.propid_counter
            self.propid_counter += 1

        self.initialized = True

        # Time to make the startup decision, COLD, WARM or HOT
        config = kwargs.pop('config')

        if config.sched_driver.startup_type == 'HOT':
            # This is the regular startup, will just return without doing anything.
            self.log.info("Start up type is HOT, no state will be read from the EFD.")
        elif config.sched_driver.startup_type != 'HOT' and \
                config.sched_driver.startup_database is None:
            raise IOError('Startup database not defined for startup type {}'.format(
                config.sched_driver.startup_type))
        elif config.sched_driver.startup_type == 'WARM':
            raise NotImplementedError('Warm start not implemented yet.')
        elif config.sched_driver.startup_type == 'COLD':
            import pandas as pd
            import sqlite3

            self.log.info('Running cold start from {}'.format(config.sched_driver.startup_database))

            conn = sqlite3.connect(config.sched_driver.startup_database)
            df = pd.read_sql_query('''select ObsHistory.observationId as targetid,
     ObsHistory.night,
     ObsHistory.observationStartMJD as observation_start_mjd,
     ObsHistory.Field_fieldId as fieldid,
     ObsHistory.filter,
     ObsHistory.ra,
     ObsHistory.dec,
     ObsHistory.angle as ang,
     ObsHistory.altitude as alt,
     ObsHistory.azimuth as az,
     ObsHistory.numExposures as num_exp,
     ObsHistory.visitExposureTime as exp_time,
     ObsHistory.airmass,
     ObsHistory.skyBrightness as sky_brightness,
     ObsHistory.cloud,
     ObsHistory.seeingFwhmGeom as seeing_fwhm_geom,
     ObsHistory.seeingFwhmEff as seeing_fwhm_eff,
     ObsHistory.fiveSigmaDepth as five_sigma_depth,
     ObsHistory.moonAlt as moon_alt,
     ObsHistory.sunAlt as sun_alt,
     ObsHistory.slew_time as slewtime,
     ObsHistory.note,
     ObsProposalHistory.propHistId as propid_list from ObsHistory join ObsProposalHistory on 
    "ObsHistory"."observationId" = "ObsProposalHistory"."ObsHistory_observationId";''',
                                   conn)

            self.log.debug('Found {} observations on database.'.format(len(df)))

            from lsst.ts.observatory.model import Observation

            list_observations = []
            for iobs in range(len(df)):
                obs_dict = df.iloc[iobs].to_dict()
                obs_dict['note'] += 'coldstart'
                obs_dict['propid_list'] = [obs_dict['propid_list'], ]
                list_observations.append(Observation.make_copy(obs_dict))

            self.cold_start(list_observations)

        return survey_topology

    def cold_start(self, observations):
        """
        Configure cold start from a list of observations.

        Parameters
        ----------
        observations

        Returns
        -------

        """

        for i in range(len(observations)):
            self.last_winner_target = observations[i]
            self.register_observation(observations[i])
        self.night = observations[-1].night

    def end_survey(self):

        self.log.info("end_survey")

    def start_night(self, timestamp, night):

        self.night = night
        super(FeatureSchedulerDriver, self).start_night(timestamp, night)
        for fieldid in self.target_list.keys():
            for filtername in self.target_list[fieldid].keys():
                self.target_list[fieldid][filtername].groupix = 0

    def end_night(self, timestamp, night):

        timeprogress = (timestamp - self.start_time) / self.survey_duration_SECS
        self.log.info("end_night t=%.6f, night=%d timeprogress=%.2f%%" %
                      (timestamp, night, 100 * timeprogress))

        self.isnight = False

        self.last_winner_target = self.nulltarget
        self.deep_drilling_target = None

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
                    self.filter_to_unmount = self.observatoryModel.params.filter_removable_list[0]

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
        telemetry_stream = self.get_telemetry()

        self.scheduler.update_conditions(telemetry_stream)
        winner_target = self.scheduler.request_observation()
        if winner_target is None:
            self.log.debug('[mjd = %.3f]: No target generated by the scheduler...', telemetry_stream['mjd'])
            self.scheduler.flush_queue()
            self.last_winner_target = self.nulltarget.get_copy()
            return self.last_winner_target

        self.log.debug('winner target: %s', winner_target)
        self.scheduler_winner_target = winner_target

        hpid = _raDec2Hpid(self.sky_nside, winner_target['RA'][0], winner_target['dec'][0])

        propid = winner_target['survey_id'][0]
        filtername = winner_target['filter'][0]
        indx = self.proposal_id_dict[propid][0]
        target = self.generate_target(winner_target[0])

        self.target_list[target.fieldid] = {filtername: target}
        # self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {filtername: target}

        target.time = self.time

        target.ang_rad = self.observatoryModel.radec2altazpa(
            self.observatoryModel.dateprofile, target.ra_rad, target.dec_rad)[2]
        self.observatoryModel.current_state.ang_rad = target.ang_rad
        slewtime, slew_state = self.observatoryModel.get_slew_delay(target)

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
            self.scheduler_winner_target['rotSkyPos'] = target.ang_rad
            self.scheduler_winner_target['clouds'] = self.cloud
            self.scheduler_winner_target['sunAlt'] = telemetry_stream['sunAlt']
            self.scheduler_winner_target['moonAlt'] = telemetry_stream['moonAlt']

            target.slewtime = slewtime
            target.airmass = telemetry_stream['airmass'][hpid]
            target.sky_brightness = self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd,
                                                                   indx=[hpid],
                                                                   extrapolate=True)[filtername][0]

            self.observatoryModel2.set_state(self.observatoryState)
            self.observatoryState.ang_rad = target.ang_rad
            self.observatoryModel2.observe(target)
            target.seeing = self.seeing
            target.cloud = self.cloud

            ntime = self.observatoryModel2.current_state.time
            if ntime < self.sunrise_timestamp:
                # self.observatoryModel2.update_state(ntime)
                if self.observatoryModel2.current_state.tracking:
                    target.time = self.time
                    if self.last_winner_target.targetid == target.targetid:
                        self.last_winner_target = self.nulltarget
                        self.targetid -= 1
                    else:
                        self.last_winner_target = target.get_copy()
                else:
                    self.log.debug("select_next_target: target rejected %s",
                                   (str(target)))
                    self.log.debug("select_next_target: state rejected %s",
                                   str(self.observatoryModel2.current_state))
                    self.last_winner_target = self.nulltarget
                    self.targetid -= 1
            else:
                self.last_winner_target = self.nulltarget
                self.targetid -= 1

        else:
            self.log.debug('Fail state: %i', slew_state)
            self.log.debug('Slewtime lower than zero! (slewtime = %f)', slewtime)
            self.scheduler.flush_queue()
            self.targetid -= 1
            self.last_winner_target = self.nulltarget
            self.scheduler_winner_target = None

        self.log.debug(self.last_winner_target)
        # for propid in self.proposal_id_dict.keys():
        #     self.science_proposal_list[self.proposal_id_dict[propid][0]].winners_list = []
        # self.science_proposal_list[indx].winners_list = [target.get_copy()]

        return self.last_winner_target

    def register_observation(self, observation):
        if observation.targetid > 0:
            # FIXME: Add conversion of observation to fbs target
            self.log.debug('Adding %s', observation)
            fbs_obs = obs_to_fbsobs(observation)
            self.scheduler.add_observation(fbs_obs)
            return [self.last_winner_target]
        else:
            return []

    def update_time(self, timestamp, night):

        delta_time = timestamp-self.time
        self.time = timestamp
        self.observatoryModel.update_state(self.time)

        if not self.survey_started:
            self.start_survey(timestamp, night)

        if self.isnight:
            # if round(timestamp) >= round(self.sunrise_timestamp):
            if timestamp >= self.sunrise_timestamp:
                self.scheduler.flush_queue()  # Clean queue if night is over!
                self.end_night(timestamp, night)
        else:
            # if round(timestamp) >= round(self.sunset_timestamp):
            if timestamp >= self.sunset_timestamp:
                self.start_night(timestamp, night)

        return self.isnight

    def generate_target(self, fb_observation, tag='generate'):
        '''Takes an observation array given by the feature based scheduler and generate an appropriate OpSim target.

        :param fb_observation: numpy.array
        :return: Target
        '''

        # self.log.debug('%s: %s' % (tag, fb_observation))
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
        target.ang_rad = fb_observation['rotSkyPos']
        target.propid = propid
        target.goal = 100
        target.visits = 0
        target.progress = 0.0
        target.groupid = 1
        target.groupix = 0
        target.num_props = 1
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
        target.note = fb_observation['note']

        return target

    def append_target(self, fb_observation):
        '''Takes an observation array given by the feature based scheduler and append it to an existing OpSim target.

        :param fb_observation: numpy.array
        :return: Target
        '''
        # self.log.debug('append: %s' % fb_observation)
        # target = self.target_list[fb_observation['field_id']][fb_observation['filter']].get_copy()
        # self.targetid += 1
        # target.targetid = self.targetid
        propid = fb_observation['survey_id']
        # target.propid = propid
        # # if propid not in target.propid_list:
        #
        # target.propid_list = [propid]
        indx = self.proposal_id_dict[propid][0]
        #
        # if target.fieldid in self.science_proposal_list[indx].survey_targets_dict:
        #     self.science_proposal_list[indx].survey_targets_dict[target.fieldid][target.filter] = target
        # else:
        #     self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {target.filter: target}
        #
        # target.ra_rad = fb_observation['RA']
        # target.dec_rad = fb_observation['dec']
        # target.groupix = 0
        # self.target_list[fb_observation['field_id']][fb_observation['filter']] = target.get_copy()

        target = self.generate_target(fb_observation, 'append')
        if target.fieldid in self.science_proposal_list[indx].survey_targets_dict:
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid][target.filter] = target
        else:
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {target.filter: target}

        return target

    def get_telemetry(self):

        telemetry_stream = {}
        telemetry_stream['mjd'] = copy.copy(self.observatoryModel.dateprofile.mjd)
        telemetry_stream['night'] = copy.copy(self.night)
        telemetry_stream['lmst'] = copy.copy(self.observatoryModel.dateprofile.lst_rad*12./np.pi) % 24.

        dp = DateProfile(0, self.observatoryModel.dateprofile.location)
        mjd, _ = dp(self.sunrise_timestamp)
        telemetry_stream['next_twilight_start'] = copy.copy(dp.mjd)

        dp = DateProfile(0, self.observatoryModel.dateprofile.location)
        mjd, _ = dp(self.sunset_timestamp)
        telemetry_stream['next_twilight_end'] = copy.copy(dp.mjd)
        telemetry_stream['last_twilight_end'] = copy.copy(dp.mjd)

        # Telemetry about where the observatory is pointing and what filter it's using.
        telemetry_stream['filter'] = copy.copy(self.observatoryModel.current_state.filter)
        telemetry_stream['mounted_filters'] = copy.copy(self.observatoryModel.current_state.mountedfilters)
        telemetry_stream['telRA'] = copy.copy(np.degrees(self.observatoryModel.current_state.ra_rad))
        telemetry_stream['telDec'] = copy.copy(np.degrees(self.observatoryModel.current_state.dec_rad))
        telemetry_stream['telAlt'] = copy.copy(np.degrees(self.observatoryModel.current_state.telalt_rad))
        telemetry_stream['telAz'] = copy.copy(np.degrees(self.observatoryModel.current_state.telaz_rad))
        telemetry_stream['telRot'] = copy.copy(np.degrees(self.observatoryModel.current_state.telrot_rad))

        # What is the sky brightness over the sky (healpix map)
        telemetry_stream['skybrightness'] = copy.copy(
            self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd))

        # Find expected slewtimes over the sky.
        # The slewtimes telemetry is expected to be a healpix map of slewtimes, in the current filter.
        # There are other features that balance the filter change cost, separately.
        # (this filter aspect may be something to revisit in the future?)
        # Note that these slewtimes are -1 where the telescope is not allowed to point.
        alt, az = _approx_RaDec2AltAz(self.scheduler.ra_grid_rad,
                                      self.scheduler.dec_grid_rad,
                                      self.observatoryModel.location.latitude_rad,
                                      self.observatoryModel.location.longitude_rad,
                                      self.observatoryModel.dateprofile.mjd,
                                      self.observatoryModel.dateprofile.lst_rad * 12. / np.pi % 24.)
        current_filter = self.observatoryModel.current_state.filter

        lax_dome = self.observatoryModel.params.domaz_free_range > 0.
        telemetry_stream['slewtimes'] = copy.copy(self.observatoryModel.get_approximate_slew_delay(alt, az,
                                                                                                   current_filter,
                                                                                                   lax_dome=lax_dome))
        # What is the airmass over the sky (healpix map).

        telemetry_stream['airmass'] = copy.copy(
            self.sky_brightness.returnAirmass(self.observatoryModel.dateprofile.mjd))

        delta_t = (self.time-self.start_time)
        telemetry_stream['clouds'] = self.cloud

        fwhm_geometric, fwhm_effective = self.seeing_model.seeing_at_airmass(self.seeing,
                                                                             telemetry_stream['airmass'])

        for i, filtername in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
            telemetry_stream['FWHMeff_%s' % filtername] = copy.copy(fwhm_effective[i])  # arcsec
            telemetry_stream['FWHM_geometric_%s' % filtername] = copy.copy(fwhm_geometric[i])

        self.sky.update(self.time)

        sunMoon_info = self.sky.get_moon_sun_info(np.array([0.0]), np.array([0.0]))

        # Pretty sure these are radians
        telemetry_stream['sunAlt'] = copy.copy(np.max(sunMoon_info['sunAlt']))
        telemetry_stream['moonAlt'] = copy.copy(np.max(sunMoon_info['moonAlt']))

        telemetry_stream['moonAz'] = copy.copy(np.max(sunMoon_info['moonAz']))
        telemetry_stream['moonRA'] = copy.copy(np.max(sunMoon_info['moonRA']))
        telemetry_stream['moonDec'] = copy.copy(np.max(sunMoon_info['moonDec']))
        telemetry_stream['moonPhase'] = copy.copy(np.max(sunMoon_info['moonPhase']))

        return telemetry_stream
