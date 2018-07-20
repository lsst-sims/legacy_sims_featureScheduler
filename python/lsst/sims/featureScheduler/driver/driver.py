
import copy
import numpy as np
from importlib import import_module
import importlib.util
from lsst.sims.featureScheduler import obs_to_fbsobs
from numpy.lib.recfunctions import append_fields

from lsst.sims.ocs.configuration import Environment
from lsst.sims.ocs.configuration.instrument import Filters
from lsst.sims.ocs.kernel.time_handler import TimeHandler
from lsst.sims.skybrightness_pre import SkyModelPre
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, calcLmstLast, m5_flat_sed
from lsst.sims.seeingModel import SeeingModel

from lsst.sims.featureScheduler.driver.constants import CONFIG_NAME

from lsst.ts.observatory.model import Target
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler import stupidFast_RaDec2AltAz
from lsst.sims.featureScheduler.coldstart import coldstarter
from lsst.ts.dateloc import DateProfile
from lsst.ts.scheduler import Driver
from lsst.ts.scheduler.proposals import AreaDistributionProposal
from lsst.sims.featureScheduler.driver.proposals import FeatureBasedProposal

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
        self.obsList = []
        self.fbs_obsList = []

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

    def create_area_proposal(self, propid, name, config_dict):
        '''Override create_area_proposal from superclass.

        One call to rule them all!

        :param propid:
        :param name:
        :param config_dict:
        :return:
        '''
        if not self.initialized and name == 'WideFastDeep':
            self.initialize(config_dict)

    def create_sequence_proposal(self, propid, name, config_dict):
        pass

    def initialize(self, config_dict):
        # Load configuration from a python module.
        # TODO:
        # This can be changed later to load from a given string, I'm planning on getting the name from
        # lsst.sims.ocs.configuration.survey.general_proposals
        conf = import_module('lsst.sims.featureScheduler.driver.config')
        from lsst.ts.scheduler.kernel import Field

        self.scheduler = conf.scheduler
        self.sky_nside = conf.nside
        # self.scheduler_visit_counting_bfs = conf.scheduler_visit_counting_bfs

        # Now, configure the different proposals inside Driver
        # Get the proposal list from feature based scheduler

        survey_fields = {}
        for survey in self.scheduler.surveys:
            if 'proposals' in survey.features:
                for i, pid in enumerate(survey.features['proposals'].id.keys()):
                    # This gets names of all proposals on all surveys, overwrites repeated and stores new ones
                    self.proposal_id_dict[pid] = [0,
                                                  survey.features['proposals'].id[pid]]

            # # Now need to get fields for each proposal
            # for field in survey.fields:
            #     if field['tag'] > 0:
            #         if field['tag'] not in survey_fields.keys():
            #             # add and skip to next iteration
            #             survey_fields[field['tag']] = np.array([field])
            #             continue
            #
            #         if field['field_id'] not in survey_fields[field['tag']]['field_id']:
            #             survey_fields[field['tag']] = np.append(survey_fields[field['tag']],
            #                                                     field)

        self.log.debug('Proposal_id_dict: %s' % self.proposal_id_dict.keys())
        # self.log.debug('survey_fields: %s' % survey_fields.keys())

        for pid in self.proposal_id_dict.keys():
            self.proposal_id_dict[pid][0] = self.propid_counter
            self.propid_counter += 1

            self.log.debug('%s: %s' % (pid, self.proposal_id_dict[pid]))

            prop = FeatureBasedProposal(pid,
                                        self.proposal_id_dict[pid][1],
                                        config_dict,
                                        self.sky)

            prop.configure_constraints(self.params)
            # create proposal field list
            # prop.survey_fields = len(survey_fields[pid])
            # for field in survey_fields[pid]:
            #     prop.survey_fields_dict[field['field_id']] = Field(fieldid=field['field_id'],
            #                                                        ra_rad=field['RA'],
            #                                                        dec_rad=field['dec'],
            #                                                        gl_rad=field['gl'],
            #                                                        gb_rad=field['gb'],
            #                                                        el_rad=field['el'],
            #                                                        eb_rad=field['eb'],
            #                                                        fov_rad=field['fov_rad'])
            self.science_proposal_list.append(prop)

        self.initialized = True

    def end_survey(self):
        self.log.info("end_survey")

    def start_night(self, timestamp, night):

        self.night = night
        super(FeatureSchedulerDriver, self).start_night(timestamp, night)
        for fieldid in self.target_list.keys():
            for filtername in self.target_list[fieldid].keys():
                self.target_list[fieldid][filtername].groupix = 0

    def end_night(self, timestamp, night):
        #put the telescope in a 'parked position' by setting filter to None.
        telemetry_stream = self.get_telemetry()
        telemetry_stream['filter'] = None
        self.scheduler.update_conditions(telemetry_stream)

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
        

        self.obsList = []
        self.fbs_obsList = []

        super(FeatureSchedulerDriver, self).end_night(timestamp, night)

    def select_next_target(self):

        if not self.isnight:
            return self.nulltarget

        # Telemetry stream
        telemetry_stream = self.get_telemetry()

        self.scheduler.update_conditions(telemetry_stream)
        winner_target = self.scheduler.request_observation()
        if winner_target is None:
            self.log.debug('[mjd = %.3f]: No target generated by the scheduler...' % telemetry_stream['mjd'])
            self.scheduler.flush_queue()
            self.last_winner_target = self.nulltarget.get_copy()
            return self.last_winner_target

        self.log.debug('winner target: %s' % winner_target)
        self.scheduler_winner_target = winner_target

        hpid = _raDec2Hpid(self.sky_nside, winner_target['RA'][0], winner_target['dec'][0])

        propid = winner_target['survey_id'][0]
        filtername = winner_target['filter'][0]
        indx = self.proposal_id_dict[propid][0]
        target = self.generate_target(winner_target[0])

        self.target_list[target.fieldid] = {filtername: target}
        self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {filtername: target}

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
                    self.log.debug("select_next_target: target rejected %s" %
                                   (str(target)))
                    self.log.debug("select_next_target: state rejected %s" %
                                   str(self.observatoryModel2.current_state))
                    self.last_winner_target = self.nulltarget
                    self.targetid -= 1
            else:
                self.last_winner_target = self.nulltarget
                self.targetid -= 1

        else:
            self.log.debug('Fail state: %i' % slew_state)
            self.log.debug('Slewtime lower than zero! (slewtime = %f)' % slewtime)
            self.scheduler.flush_queue()
            self.targetid -= 1
            self.last_winner_target = self.nulltarget

        self.log.debug(self.last_winner_target)
        for propid in self.proposal_id_dict.keys():
            self.science_proposal_list[self.proposal_id_dict[propid][0]].winners_list = []
        self.science_proposal_list[indx].winners_list = [target.get_copy()]

        return self.last_winner_target

    def register_observation(self, observation, isColdStart = False):
        if isColdStart:
            self.obsList.append(copy.copy(observation))
            self.fbs_obsList.append(observation)
            self.scheduler.add_observation(observation) 
            
        elif observation.targetid > 0:
            fbsobs = obs_to_fbsobs(observation) 
            self.obsList.append(copy.copy(observation))
            self.fbs_obsList.append(fbsobs)
            self.scheduler.add_observation(fbsobs) 
            return super(FeatureSchedulerDriver, self).register_observation(observation, isColdStart)
        else:
            return []

    def cold_start(self, fileName=None):

        """Rebuilds the state of the scheduler from a list of observations"""
        self.log.info("Running coldstart (fbs)")
        obs_history = coldstarter.get_observation_history("docker_mothra_2117.db")
        self.log.info("Loaded " + str(len(obs_history)) + " observations from coldstart database.")
        for obs in obs_history:
            self.register_observation(obs, isColdStart = True)
        
        self.log.info("Coldstart finished")

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
        telemetry_stream['telAlt'] = copy.copy(np.degrees(self.observatoryModel.current_state.alt_rad))
        telemetry_stream['telAz'] = copy.copy(np.degrees(self.observatoryModel.current_state.az_rad))
        telemetry_stream['telRot'] = copy.copy(np.degrees(self.observatoryModel.current_state.rot_rad))

        # What is the sky brightness over the sky (healpix map)
        telemetry_stream['skybrightness'] = copy.copy(
            self.sky_brightness.returnMags(self.observatoryModel.dateprofile.mjd))

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

        telemetry_stream['slewtimes'] = copy.copy(self.observatoryModel.get_approximate_slew_delay(alt, az,
                                                                                         current_filter))
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

        sunMoon_info = self.sky_brightness.returnSunMoon(self.observatoryModel.dateprofile.mjd)
        # Pretty sure these are radians
        telemetry_stream['sunAlt'] = copy.copy(np.max(sunMoon_info['sunAlt']))
        telemetry_stream['moonAlt'] = copy.copy(np.max(sunMoon_info['moonAlt']))

        return telemetry_stream
