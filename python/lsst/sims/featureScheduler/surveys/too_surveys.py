import numpy as np
from lsst.sims.featureScheduler.surveys import Blob_survey, BaseSurvey
import healpy as hp
import copy


class ToO_master(BaseSurvey):
    """
    A target of opportunity class. Every time a new ToO comes in, it will spawn a new sub-survey.
    """

    def __init__(self, example_ToO_survey):
        self.example_ToO_survey = example_ToO_survey
        self.surveys = []
        self.highest_reward = -np.inf

    def add_observation(self, observation, **kwargs):
        if len(self.surveys) > 0:
            for key in self.surveys:
                self.surveys[key].add_observation(observation)

    def _spawn_new_survey(self, too):
        """Create a new survey object for a ToO we haven't seen before.

        Parameters
        ----------
        too : lsst.sims.featureScheduler.utils.TargetoO object
        """
        new_survey = copy.deepcopy(self.example_ToO_survey)
        new_survey.set_id(too.id)
        return new_survey

    def _check_survey_list(self, conditions):
        """There is a current ToO in the conditions.
        """

        running_ids = [survey.too_id for survey in self.surveys]
        current_ids = [too.id for too in conditions.targets_of_opportunity]

        # delete any ToO surveys that are no longer relevant
        self.surveys = [survey for survey in self.surveys if survey.too_id in current_ids]

        # Spawn new surveys that are needed
        new_surveys = []
        for too in conditions.targets_of_opportunity:
            if too.id not in running_ids:
                new_surveys.append(self._spawn_new_survey(too))
        self.surveys.extend(new_surveys)

    def calc_reward_function(self, conditions):
        # Catch if a new ToO has happened
        if conditions.targets_of_opportunity is not None:
            self._check_survey_list(conditions)

        if len(self.surveys) > 0:
            rewards = [survey.calc_reward_function(conditions) for survey in self.surveys]
            self.reward = np.max(rewards)
            self.highest_reward = np.min(np.where(rewards == self.reward))
        else:
            self.reward = -np.inf
            self.highest_reward = None
        return self.reward

    def generate_observations(self, conditions):
        if self.reward > -np.inf:
            return self.surveys[self.highest_reward].generate_observations(conditions)


class ToO_survey(Blob_survey):
    """Survey class to catch incoming target of opportunity anouncements and try to observe them.

    The idea is that we can dynamically update the target footprint basis function, and add new features as more ToOs come in.

    Parameters
    ----------
    too_id : int (None)
        A unique integer ID for the ToO getting observed
    """
    def __init__(self, basis_functions, basis_weights,
                 filtername1='r', filtername2=None,
                 slew_approx=7.5, filter_change_approx=140.,
                 read_approx=2., exptime=30., nexp=2,
                 ideal_pair_time=22., min_pair_time=15.,
                 search_radius=30., alt_max=85., az_range=90.,
                 flush_time=30.,
                 smoothing_kernel=None, nside=None,
                 dither=True, seed=42, ignore_obs=None,
                 survey_note='ToO', detailers=None, camera='LSST',
                 too_id=None):
        super(ToO_survey, self).__init__(basis_functions=basis_functions, basis_weights=basis_weights,
                                         filtername1=filtername1, fitlername2=filtername2, slew_approx=slew_approx,
                                         filter_change_approx=filter_change_approx, read_approx=read_approx, exptime=exptime,
                                         nexp=nexp, ideal_pair_time=ideal_pair_time, min_pair_time=min_pair_time, search_radius=search_radius,
                                         alt_max=alt_max, az_range=az_range, flush_time=flush_time, smoothing_kernel=smoothing_kernel, nside=nside,
                                         dither=dither, seed=seed, ignore_obs=ignore_obs, survey_note=survey_note, detailers=detailers, camera=camera)
        # Include the ToO id in the note
        self.survey_note_base = self.survey_note
        self.set_id(too_id)

    def set_id(self, newid):
        """Set the id
        """
        self.to_id = newid
        self.survey_note = self.survey_note_base + ', ' + str(newid)

    def generate_observations_rough(self, conditions):
        # Always spin the tesselation before generating a new block.
        if self.dither:
            self._spin_fields()
        result = super(ToO_survey, self).generate_observations_rough(conditions)
        return result
