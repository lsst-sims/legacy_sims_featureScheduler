from builtins import object
from builtins import range
from builtins import str
import os
import math
import numpy
from numpy.lib.recfunctions import append_fields
import logging
import healpy as hp
from operator import itemgetter

from lsst.ts.astrosky.model import AstronomicalSkyModel
from lsst.ts.dateloc import ObservatoryLocation
from lsst.ts.observatory.model import ObservatoryModel
from lsst.ts.observatory.model import ObservatoryState
from lsst.ts.observatory.model import Target
from lsst.ts.scheduler.setup import EXTENSIVE, WORDY
from lsst.ts.scheduler.kernel import read_conf_file
from lsst.ts.scheduler.kernel import Field
from lsst.ts.scheduler.proposals import ScriptedProposal
from lsst.ts.scheduler.proposals import AreaDistributionProposal, TimeDistributionProposal
from lsst.ts.scheduler.fields import FieldsDatabase

import lsst.sims.featureScheduler as fs

__all__ = ["Driver"]

class DriverParameters(object):

    def __init__(self):

        self.coadd_values = False
        self.timecost_weight = 0.0
        self.timecost_dc = 0.0
        self.timecost_dt = 0.0
        self.timecost_k = 0.0
        self.night_boundary = 0.0
        self.ignore_sky_brightness = False
        self.ignore_airmass = False
        self.ignore_clouds = False
        self.ignore_seeing = False
        self.new_moon_phase_threshold = 0.0

    def configure(self, confdict):

        self.coadd_values = confdict["ranking"]["coadd_values"]
        self.time_balancing = confdict["ranking"]["time_balancing"]

        tmax = confdict["ranking"]["timecost_time_max"]
        tref = confdict["ranking"]["timecost_time_ref"]
        cref = float(confdict["ranking"]["timecost_cost_ref"])

        self.timecost_tmax = tmax
        self.timecost_tref = tref
        self.timecost_cref = cref

        self.timecost_dc = cref * (tmax - tref) / (tref - cref * tmax)
        self.timecost_dt = -tmax * (self.timecost_dc + 1.0)
        self.timecost_k = self.timecost_dc * self.timecost_dt
        self.timecost_weight = confdict["ranking"]["timecost_weight"]
        self.filtercost_weight = confdict["ranking"]["filtercost_weight"]
        self.propboost_weight = confdict["ranking"]["propboost_weight"]

        self.night_boundary = confdict["constraints"]["night_boundary"]
        self.ignore_sky_brightness = confdict["constraints"]["ignore_sky_brightness"]
        self.ignore_airmass = confdict["constraints"]["ignore_airmass"]
        self.ignore_clouds = confdict["constraints"]["ignore_clouds"]
        self.ignore_seeing = confdict["constraints"]["ignore_seeing"]
        self.new_moon_phase_threshold = confdict["darktime"]["new_moon_phase_threshold"]

class Driver(object):
    def __init__(self):

        self.log = logging.getLogger("featureSchedulerDriver")

        self.params = DriverParameters()
        self.location = ObservatoryLocation()

        self.observatoryModel = ObservatoryModel(self.location, WORDY)
        self.observatoryModel2 = ObservatoryModel(self.location, WORDY)
        self.observatoryState = ObservatoryState()

        self.sky = AstronomicalSkyModel(self.location)

        self.db = FieldsDatabase()

        self.build_fields_dict()

        self.propid_counter = 0
        self.science_proposal_list = []

        self.start_time = 0.0
        self.time = 0.0
        self.targetid = 0
        self.survey_started = False
        self.isnight = False
        self.sunset_timestamp = 0.0
        self.sunrise_timestamp = 0.0
        self.survey_duration_DAYS = 0.0
        self.survey_duration_SECS = self.survey_duration_DAYS * 24 * 60 * 60.0
        self.darktime = False
        self.mounted_filter = ""
        self.unmounted_filter = ""
        self.midnight_moonphase = 0.0

        self.nulltarget = Target()
        self.nulltarget.targetid = -1
        self.nulltarget.num_exp = 1
        self.nulltarget.exp_times = [0.0]
        self.nulltarget.num_props = 1
        self.nulltarget.propid_list = [0]
        self.nulltarget.need_list = [0.0]
        self.nulltarget.bonus_list = [0.0]
        self.nulltarget.value_list = [0.0]
        self.nulltarget.propboost_list = [1.0]
        self.last_winner_target = self.nulltarget.get_copy()
        self.deep_drilling_target = None

        self.need_filter_swap = False
        self.filter_to_unmount = ""
        self.filter_to_mount = ""

        self.cloud = 0.0
        self.seeing = 0.0


        self.configure_survey(None)

    def configure_survey(self, survey_conf_file):

        # prop_conf_path = os.path.dirname(survey_conf_file)
        # confdict = read_conf_file(survey_conf_file)

        # self.configure_duration(confdict["survey"]["survey_duration"])

        self.propid_counter = 0
        self.science_proposal_list = []

        # Fixme: hardcoded configuration
        target_maps = fs.standard_goals()
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        surveys = []

        for filtername in filters:
            bfs = []
            bfs.append(fs.M5_diff_basis_function(filtername=filtername))
            bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                    target_map=target_maps[filtername],
                                                    out_of_bounds_val=hp.UNSEEN))

            bfs.append(fs.Zenith_mask_basis_function(maxAlt=78., penalty=-100))
            bfs.append(fs.Slewtime_basis_function(filtername=filtername))
            bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

            weights = numpy.array([3.0, 0.4, 1., 2., 3.])
            surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=True))

        surveys.append(fs.Pairs_survey_scripted([], [], ignore_obs='DD'))
        dd_survey = fs.Scripted_survey([], [])
        names = ['RA', 'dec', 'mjd', 'filter']
        types = [float, float, float, '|1U']
        observations = numpy.loadtxt('minion_dd.csv', skiprows=1, dtype=list(zip(names, types)), delimiter=',')
        exptimes = numpy.zeros(observations.size)
        exptimes.fill(30.)
        observations = append_fields(observations, 'exptime', exptimes)
        nexp = numpy.zeros(observations.size)
        nexp.fill(2)
        observations = append_fields(observations, 'nexp', nexp)
        notes = numpy.zeros(observations.size, dtype='|2U')
        notes.fill('DD')
        observations = append_fields(observations, 'note', notes)
        dd_survey.set_script(observations)
        surveys.append(dd_survey)

        self.scheduler = fs.Core_scheduler(surveys)


    def configure_duration(self, survey_duration):

        self.survey_duration_DAYS = survey_duration
        self.survey_duration_SECS = survey_duration * 24 * 60 * 60.0

    def configure(self, confdict):

        self.params.configure(confdict)
        self.log.log(WORDY,
                     "configure: coadd_values=%s" % (self.params.coadd_values))
        self.log.log(WORDY,
                     "configure: time_balancing=%s" % (self.params.time_balancing))
        self.log.log(WORDY,
                     "configure: timecost_dc=%.3f" % (self.params.timecost_dc))
        self.log.log(WORDY,
                     "configure: timecost_dt=%.3f" % (self.params.timecost_dt))
        self.log.log(WORDY,
                     "configure: timecost_k=%.3f" % (self.params.timecost_k))
        self.log.log(WORDY,
                     "configure: timecost_weight=%.3f" % (self.params.timecost_weight))
        self.log.log(WORDY,
                     "configure: night_boundary=%.1f" % (self.params.night_boundary))
        self.log.log(WORDY,
                     "configure: ignore_sky_brightness=%s" % (self.params.ignore_sky_brightness))
        self.log.log(WORDY,
                     "configure: ignore_airmass=%s" % (self.params.ignore_airmass))
        self.log.log(WORDY,
                     "configure: ignore_clouds=%s" % (self.params.ignore_clouds))
        self.log.log(WORDY,
                     "configure: ignore_seeing=%s" % (self.params.ignore_seeing))
        self.log.log(WORDY,
                     "configure: new_moon_phase_threshold=%.2f" % (self.params.new_moon_phase_threshold))

        for prop in self.science_proposal_list:
            prop.configure_constraints(self.params)

    def configure_location(self, confdict):

        self.location.configure(confdict)
        self.observatoryModel.location.configure(confdict)
        self.observatoryModel2.location.configure(confdict)
        self.sky.update_location(self.location)

    def configure_observatory(self, confdict):

        self.observatoryModel.configure(confdict)
        self.observatoryModel2.configure(confdict)

    def configure_telescope(self, confdict):

        self.observatoryModel.configure_telescope(confdict)
        self.observatoryModel2.configure_telescope(confdict)

    def configure_rotator(self, confdict):

        self.observatoryModel.configure_rotator(confdict)
        self.observatoryModel2.configure_rotator(confdict)

    def configure_dome(self, confdict):

        self.observatoryModel.configure_dome(confdict)
        self.observatoryModel2.configure_dome(confdict)

    def configure_optics(self, confdict):

        self.observatoryModel.configure_optics(confdict)
        self.observatoryModel2.configure_optics(confdict)

    def configure_camera(self, confdict):

        self.observatoryModel.configure_camera(confdict)
        self.observatoryModel2.configure_camera(confdict)

    def configure_slew(self, confdict):

        self.observatoryModel.configure_slew(confdict)
        self.observatoryModel2.configure_slew(confdict)

    def configure_park(self, confdict):

        self.observatoryModel.configure_park(confdict)
        self.observatoryModel2.configure_park(confdict)

    def create_area_proposal(self, propid, name, config_dict):

        self.propid_counter += 1
        area_prop = AreaDistributionProposal(propid, name, config_dict, self.sky)
        area_prop.configure_constraints(self.params)
        self.science_proposal_list.append(area_prop)

    def create_sequence_proposal(self, propid, name, config_dict):

        self.propid_counter += 1
        seq_prop = TimeDistributionProposal(propid, name, config_dict, self.sky)
        seq_prop.configure_constraints(self.params)
        self.science_proposal_list.append(seq_prop)

    def build_fields_dict(self):

        sql = "select * from Field"
        res = self.db.query(sql)

        self.fields_dict = {}
        for row in res:
            field = Field()
            fieldid = row[0]
            field.fieldid = fieldid
            field.fov_rad = math.radians(row[1])
            field.ra_rad = math.radians(row[2])
            field.dec_rad = math.radians(row[3])
            field.gl_rad = math.radians(row[4])
            field.gb_rad = math.radians(row[5])
            field.el_rad = math.radians(row[6])
            field.eb_rad = math.radians(row[7])
            self.fields_dict[fieldid] = field
            self.log.log(EXTENSIVE, "buildFieldsTable: %s" % (self.fields_dict[fieldid]))
        self.log.info("buildFieldsTable: %d fields" % (len(self.fields_dict)))

    def get_fields_dict(self):

        return self.fields_dict

    def start_survey(self, timestamp, night):

        self.start_time = timestamp
        self.log.info("start_survey t=%.6f" % timestamp)

        self.survey_started = True
        for prop in self.science_proposal_list:
            prop.start_survey()

        self.sky.update(timestamp)
        (sunset, sunrise) = self.sky.get_night_boundaries(self.params.night_boundary)
        self.log.debug("start_survey sunset=%.6f sunrise=%.6f" % (sunset, sunrise))
        # if round(sunset) <= round(timestamp) < round(sunrise):
        if sunset <= timestamp < sunrise:
            self.start_night(timestamp, night)

        self.sunset_timestamp = sunset
        self.sunrise_timestamp = sunrise

    def end_survey(self):

        self.log.info("end_survey")

        for prop in self.science_proposal_list:
            prop.end_survey()

    def start_night(self, timestamp, night):

        timeprogress = (timestamp - self.start_time) / self.survey_duration_SECS
        self.log.info("start_night t=%.6f, night=%d timeprogress=%.2f%%" %
                      (timestamp, night, 100 * timeprogress))

        self.isnight = True

        for prop in self.science_proposal_list:
            prop.start_night(timestamp, self.observatoryModel.current_state.mountedfilters, night)

    def end_night(self, timestamp, night):

        timeprogress = (timestamp - self.start_time) / self.survey_duration_SECS
        self.log.info("end_night t=%.6f, night=%d timeprogress=%.2f%%" %
                      (timestamp, night, 100 * timeprogress))

        self.isnight = False

        self.last_winner_target = self.nulltarget
        self.deep_drilling_target = None

        total_filter_visits_dict = {}
        total_filter_goal_dict = {}
        total_filter_progress_dict = {}
        for prop in self.science_proposal_list:
            prop.end_night(timestamp)
            filter_visits_dict = {}
            filter_goal_dict = {}
            filter_progress_dict = {}
            for filter in self.observatoryModel.filters:
                if filter not in total_filter_visits_dict:
                    total_filter_visits_dict[filter] = 0
                    total_filter_goal_dict[filter] = 0
                filter_visits_dict[filter] = prop.get_filter_visits(filter)
                filter_goal_dict[filter] = prop.get_filter_goal(filter)
                filter_progress_dict[filter] = prop.get_filter_progress(filter)
                total_filter_visits_dict[filter] += filter_visits_dict[filter]
                total_filter_goal_dict[filter] += filter_goal_dict[filter]
                self.log.debug("end_night propid=%d name=%s filter=%s progress=%.2f%%" %
                               (prop.propid, prop.name, filter, 100 * filter_progress_dict[filter]))
        for filter in self.observatoryModel.filters:
            if total_filter_goal_dict[filter] > 0:
                total_filter_progress_dict[filter] = \
                    float(total_filter_visits_dict[filter]) / total_filter_goal_dict[filter]
            else:
                total_filter_progress_dict[filter] = 0.0
            self.log.info("end_night filter=%s progress=%.2f%%" %
                          (filter, 100 * total_filter_progress_dict[filter]))

        previous_midnight_moonphase = self.midnight_moonphase
        self.sky.update(timestamp)
        (sunset, sunrise) = self.sky.get_night_boundaries(self.params.night_boundary)
        self.log.debug("end_night sunset=%.6f sunrise=%.6f" % (sunset, sunrise))

        self.sunset_timestamp = sunset
        self.sunrise_timestamp = sunrise
        next_midnight = (sunset + sunrise) / 2
        self.sky.update(next_midnight)
        info = self.sky.get_moon_sun_info(numpy.array([0.0]), numpy.array([0.0]))
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
                        if total_filter_progress_dict[filter] > max_progress:
                            self.filter_to_unmount = filter
                            max_progress = total_filter_progress_dict[filter]
                    self.darktime = True
            else:
                self.log.info("end_night bright time waxing")

        if self.need_filter_swap:
            self.log.debug("end_night filter swap %s=>cam=>%s" %
                           (self.filter_to_mount, self.filter_to_unmount))

    def swap_filter(self, filter_to_unmount, filter_to_mount):

        self.log.info("swap_filter swap %s=>cam=>%s" % (filter_to_mount, filter_to_unmount))

        self.observatoryModel.swap_filter(filter_to_unmount)

        self.unmounted_filter = filter_to_unmount
        self.mounted_filter = filter_to_mount

        return

    def update_time(self, timestamp, night):

        self.time = timestamp
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

    def get_need_filter_swap(self):

        return (self.need_filter_swap, self.filter_to_unmount, self.filter_to_mount)

    def update_internal_conditions(self, observatory_state, night):

        if observatory_state.unmountedfilters != self.observatoryModel.current_state.unmountedfilters:
            unmount = observatory_state.unmountedfilters[0]
            mount = self.observatoryModel.current_state.unmountedfilters[0]
            self.swap_filter(unmount, mount)
            for prop in self.science_proposal_list:
                prop.start_night(observatory_state.time, observatory_state.mountedfilters, night)

        self.time = observatory_state.time
        self.observatoryModel.set_state(observatory_state)
        self.observatoryState.set(observatory_state)

    def update_external_conditions(self, cloud, seeing):

        self.cloud = cloud
        self.seeing = seeing

        return

    def select_next_target(self):
        if not self.isnight:
            return self.nulltarget

        self.scheduler.update_conditions(self.observatoryModel.return_status())
        self.last_winner_target = self.scheduler.request_observation()

        return self.last_winner_target

    def register_observation(self, observation):

        target_list = []
        if observation.targetid > 0:
            if self.observation_fulfills_target(observation, self.last_winner_target):
                observation = self.last_winner_target
            else:
                self.log.info("register_observation: unexpected observation %s" % str(observation))

            for prop in self.science_proposal_list:
                target = prop.register_observation(observation)
                if target is not None:
                    target_list.append(target)

        self.last_observation = observation.get_copy()
        if self.last_observation.is_deep_drilling and (self.last_observation.remaining_dd_visits > 1):
            self.deep_drilling_target = self.last_observation
        else:
            self.deep_drilling_target = None

        return target_list

    def compute_slewtime_cost(self, slewtime):

        cost = (self.params.timecost_k / (slewtime + self.params.timecost_dt) -
                self.params.timecost_dc - self.params.timecost_cref) / (1.0 - self.params.timecost_cref)
        #cost = self.params.timecost_k / (slewtime + self.params.timecost_dt) - self.params.timecost_dc

        return cost

    def compute_filterchange_cost(self):

        t = self.observatoryModel.get_delta_last_filterchange()
        T = self.observatoryModel.params.filter_max_changes_avg_interval
        if t < T:
            cost = 1.0 - t / T
        else:
            cost = 0.0

        return cost

    def observation_fulfills_target(self, observ, target):

        return (observ.fieldid == target.fieldid) and (observ.filter == target.filter)
