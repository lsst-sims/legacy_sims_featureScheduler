
import numpy as np
from numpy.lib.recfunctions import append_fields
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.driver import Driver
import healpy as hp
import logging

__all__ = ["FeatureSchedulerDriver"]


class FeatureSchedulerDriver(Driver):

    def __init__(self):

        Driver.__init__(self)
        self.log = logging.getLogger("featureSchedulerDriver")

    def configure_survey(self, survey_conf_file):

        self.propid_counter = 0
        self.science_proposal_list = []

        # Fixme: hardcoded configuration
        target_maps = fs.standard_goals()
        filters = self.observatoryModel.filters
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

            weights = np.array([3.0, 0.4, 1., 2., 3.])
            surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=True))

        self.scheduler = fs.Core_scheduler(surveys)

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
                               (prop.propid, prop.name, filter, filter_visits_dict[filter]))

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

        self.scheduler.update_conditions(self.observatoryModel.return_status())
        self.last_winner_target = self.scheduler.request_observation()

        return self.last_winner_target
