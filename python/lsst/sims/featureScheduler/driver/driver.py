
import copy
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
from lsst.sims.featureScheduler.driver.proposals import FeatureBasedProposal

import logging

__all__ = ["FeatureSchedulerDriver"]


class FeatureSchedulerDriver(Driver):

    def __init__(self):

        Driver.__init__(self)
        self.log = logging.getLogger("featureSchedulerDriver")

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

        self.time_distribution = False

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

        One call to rule them all!

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
        proposal_type_dict = dict()
        for prop in self.scheduler.surveys:
            if prop.survey_type == 'AreaDistributionProposal':
                for i, pid in enumerate(prop.basis_functions[self.scheduler_visit_counting_bfs].id_list):
                    self.proposal_id_dict[pid] = [0,
                                                  prop.basis_functions[self.scheduler_visit_counting_bfs].name_list[i]]
                    proposal_type_dict[pid] = prop.survey_type
            elif prop.survey_type == 'TimeDistributionProposal':
                pid = prop.survey_name.split(":")[0]
                proposal_type_dict[prop.survey_id] = prop.survey_type
                self.proposal_id_dict[prop.survey_id] = [0,
                                                         pid]

        for pid in self.proposal_id_dict.keys():
            self.proposal_id_dict[pid][0] = self.propid_counter
            self.propid_counter += 1

            self.log.debug('%s: %s - %s' % (pid, self.proposal_id_dict[pid], proposal_type_dict[pid]))

            if proposal_type_dict[pid] == 'AreaDistributionProposal':
                area_prop = FeatureBasedProposal(pid,
                                                 self.proposal_id_dict[pid][1],
                                                 config_dict,
                                                 self.sky)
            else:
                area_prop = FeatureBasedProposal(pid,
                                                 self.proposal_id_dict[pid][1],
                                                 config_dict,
                                                 self.sky)

            area_prop.configure_constraints(self.params)
            self.science_proposal_list.append(area_prop)

    def end_survey(self):

        self.log.info("end_survey")

    def start_night(self, timestamp, night):

        self.night = night
        super(FeatureSchedulerDriver, self).start_night(timestamp, night)
        for fieldid in self.target_list.keys():
            for filtername in self.target_list[fieldid].keys():
                self.target_list[fieldid][filtername].groupix = 0

    def select_next_target(self):

        if not self.isnight:
            return self.nulltarget

        # Telemetry stream
        telemetry_stream = self.get_telemetry()

        self.scheduler.update_conditions(telemetry_stream)
        winner_target = self.scheduler.request_observation()
        self.log.debug('winner target: %s' % winner_target)
        self.scheduler_winner_target = winner_target

        hpid = _raDec2Hpid(self.sky_nside, winner_target['RA'][0], winner_target['dec'][0])

        propid = winner_target['survey_id'][0]
        filtername = winner_target['filter'][0]
        indx = self.proposal_id_dict[propid][0]

        if winner_target['field_id'][0] in self.target_list:
            if winner_target['filter'][0] in self.target_list[winner_target['field_id'][0]]:
                target = self.append_target(winner_target[0])
            else:
                target = self.generate_target(winner_target[0])
                self.target_list[target.fieldid][filtername] = target
                if target.fieldid in self.science_proposal_list[indx].survey_targets_dict:
                    self.science_proposal_list[indx].survey_targets_dict[target.fieldid][filtername] = target
                else:
                    self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {filtername: target}
        else:
            target = self.generate_target(winner_target[0])
            self.target_list[target.fieldid] = {filtername: target}
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {filtername: target}
        target.time = self.time

        if target.targetid != self.last_winner_target.targetid:
            self.observatoryModel.current_state.telrot_rad = 0.
        slewtime = self.observatoryModel.get_slew_delay(target, use_telrot=True)

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
            self.observatoryState.telrot_rad = 0.
            self.observatoryModel2.observe(target, use_telrot=True)
            target.seeing = self.seeing
            target.cloud = self.cloud

            ntime = self.observatoryModel2.current_state.time
            if ntime < self.sunrise_timestamp:
                # self.observatoryModel2.update_state(ntime)
                if self.observatoryModel2.current_state.tracking:
                    target.time = self.time
                    if self.last_winner_target.targetid == target.targetid:
                        self.last_winner_target = self.nulltarget
                    else:
                        self.last_winner_target = target.get_copy()
                else:
                    self.log.debug("select_next_target: target rejected %s" %
                                   (str(target)))
                    self.log.debug("select_next_target: state rejected %s" %
                                   str(self.observatoryModel2.current_state))
                    self.last_winner_target = self.nulltarget
            else:
                self.last_winner_target = self.nulltarget

        else:
            self.log.debug('Slewtime lower than zero! (slewtime = %f)' % slewtime)
            self.scheduler.flush_queue()
            self.targetid -= 1
            self.last_winner_target = self.nulltarget

        self.log.debug(self.last_winner_target)
        for propid in self.proposal_id_dict.keys():
            self.science_proposal_list[self.proposal_id_dict[propid][0]].winners_list = []
        self.science_proposal_list[indx].winners_list = [target.get_copy()]

        return self.last_winner_target

    def register_observation(self, observation):
        if observation.targetid > 0:
            self.scheduler.add_observation(self.scheduler_winner_target)

            return super(FeatureSchedulerDriver, self).register_observation(observation)
        else:
            return []

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

        self.log.debug('generate: %s' % fb_observation)
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
        target.groupid = 1
        target.groupix = 0
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
        self.log.debug('append: %s' % fb_observation)
        target = self.target_list[fb_observation['field_id']][fb_observation['filter']].get_copy()
        self.targetid += 1
        target.targetid = self.targetid
        propid = fb_observation['survey_id']
        target.propid = propid
        # if propid not in target.propid_list:

        target.propid_list = [propid]
        indx = self.proposal_id_dict[propid][0]

        if target.fieldid in self.science_proposal_list[indx].survey_targets_dict:
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid][target.filter] = target
        else:
            self.science_proposal_list[indx].survey_targets_dict[target.fieldid] = {target.filter: target}

        target.ra_rad = fb_observation['RA']
        target.dec_rad = fb_observation['dec']
        target.groupix = 0
        self.target_list[fb_observation['field_id']][fb_observation['filter']] = target.get_copy()

        return target

    def get_telemetry(self):

        telemetry_stream = {}
        telemetry_stream['mjd'] = copy.copy(self.observatoryModel.dateprofile.mjd)
        telemetry_stream['night'] = copy.copy(self.night)
        telemetry_stream['lmst'] = copy.copy(self.observatoryModel.dateprofile.lst_rad*12./np.pi)

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
        telemetry_stream['clouds'] = copy.copy(self.cloud_model.get_cloud(int(delta_t)))

        for filtername in ['u', 'g', 'r', 'i', 'z', 'y']:
            fwhm_500, fwhm_geometric, fwhm_effective = self.seeing_model.calculate_seeing(delta_t, filtername,
                                                                                          telemetry_stream['airmass'])
            telemetry_stream['FWHMeff_%s' % filtername] = copy.copy(fwhm_effective)  # arcsec
            telemetry_stream['FWHM_geometric_%s' % filtername] = copy.copy(fwhm_geometric)

        sunMoon_info = self.sky_brightness.returnSunMoon(self.observatoryModel.dateprofile.mjd)
        # Pretty sure these are radians
        telemetry_stream['sunAlt'] = copy.copy(np.max(sunMoon_info['sunAlt']))
        telemetry_stream['moonAlt'] = copy.copy(np.max(sunMoon_info['moonAlt']))

        return telemetry_stream