import numpy as np
from lsst.sims.featureScheduler.surveys import Blob_survey, BaseSurvey
import lsst.sims.featureScheduler.basis_functions as basis_functions
from lsst.sims.featureScheduler.utils import empty_observation
import healpy as hp


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
        pass

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

    The idea is that we can dynamically update the target footprint basis fucntion, and add new features as more ToOs come in.
    """

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        return result

    def calc_reward_function(self, conditions):
        self.reward_checked = True
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value*weight

            if np.any(np.isinf(self.reward)):
                self.reward = np.inf
        else:
            # If not feasable, negative infinity reward
            self.reward = -np.inf
        if self.smoothing_kernel is not None:
            self.smooth_reward()
            return self.reward_smooth
        else:
            return self.reward

    def generate_observations_rough(self, conditions):

        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields()
            self.night = conditions.night.copy()

        # XXX Use self.reward to decide what to observe.
        return None