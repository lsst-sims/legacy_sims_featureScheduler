
import numpy as np
from numpy.lib.recfunctions import append_fields
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.driver import Driver

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

            weights = np.array([3.0, 0.4, 1., 2., 3.])
            surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=True))

        surveys.append(fs.Pairs_survey_scripted([], [], ignore_obs='DD'))
        dd_survey = fs.Scripted_survey([], [])
        names = ['RA', 'dec', 'mjd', 'filter']
        types = [float, float, float, '|1U']
        observations = np.loadtxt('minion_dd.csv', skiprows=1, dtype=list(zip(names, types)), delimiter=',')
        exptimes = np.zeros(observations.size)
        exptimes.fill(30.)
        observations = append_fields(observations, 'exptime', exptimes)
        nexp = np.zeros(observations.size)
        nexp.fill(2)
        observations = append_fields(observations, 'nexp', nexp)
        notes = np.zeros(observations.size, dtype='|2U')
        notes.fill('DD')
        observations = append_fields(observations, 'note', notes)
        dd_survey.set_script(observations)
        surveys.append(dd_survey)

        self.scheduler = fs.Core_scheduler(surveys)

    def select_next_target(self):
        if not self.isnight:
            return self.nulltarget

        self.scheduler.update_conditions(self.observatoryModel.return_status())
        self.last_winner_target = self.scheduler.request_observation()

        return self.last_winner_target
