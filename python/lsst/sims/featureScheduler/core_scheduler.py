from __future__ import absolute_import
from builtins import object
import numpy as np
from .utils import hp_in_lsst_fov, set_default_nside

default_nside = set_default_nside()


class Core_scheduler(object):
    """
    
    """

    def __init__(self, surveys, nside=default_nside, camera='LSST'):
        """

        """
        # initialize a queue of observations to request
        self.queue = []
        self.surveys = surveys
        self.nside = nside
        self.conditions = None
        # Should just make camera a class that takes a pointing and returns healpix indices
        if camera == 'LSST':
            self.pointing2hpindx = hp_in_lsst_fov(nside=nside)
        else:
            raise ValueError('camera %s not implamented' % camera)

    def flush_queue(self):
        """"
        Like it sounds, clear any currently queued desired observations.
        """
        self.queue = []

    def add_observation(self, observation):
        """
        Record a completed observation and update features accordingly.

        Parameters
        ----------
        observation : dict-like
            An object that contains the relevant information about a
            completed observation (e.g., mjd, ra, dec, filter, rotation angle, etc)
        """

        # Find the healpixel centers that are included in an observation
        indx = self.pointing2hpindx(observation['RA'], observation['dec'])
        for survey in self.surveys:
            survey.add_observation(observation, indx=indx)

    def update_conditions(self, conditions):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position, loaded filters, cloud-mask, etc)
        """
        # Add the current queue and scheduled queue to the conditions
        conditions['queue'] = self.queue

        for survey in self.surveys:
            survey.update_conditions(conditions)
        self.conditions = conditions

    def request_observation(self):
        """
        Ask the scheduler what it wants to observe next
        """
        if len(self.queue) == 0:
            self._fill_queue()
        result = self.queue.pop(0)
        return result

    def _fill_queue(self):
        """
        Compute reward function for each survey and fill the observing queue with the
        observations of highest reward.
        """
        rewards = []
        for survey in self.surveys:
            rewards.append(np.max(survey.calc_reward_function()))

        # Take a min here, so the surveys will be executed in the order they are
        # entered if there is a tie.
        good = np.min(np.where(rewards == np.max(rewards)))

        # Survey return list of observations
        result = self.surveys[good]()
        self.queue = result


class Core_scheduler_cost(Core_scheduler):
    """
    Because some people really like minimizing cost rather than optimizing reward.
    """

    def _fill_queue(self):
        """
        Compute cost function for each survey and fill the observing queue with the
        observations of lowest cost.
        """
        costs = []
        for survey in self.surveys:
            costs.append(survey.calc_cost_function())

        # Take a min here, so the surveys will be executed in the order they are
        # entered if there is a tie.
        good = np.min(np.where(costs == np.min(costs)))

        # Survey could return list of observations, or a survey
        result = self.surveys[good]()

        if isinstance(result, list):
            self.queue = result
        else:
            self.scripted_surveys.append(result)
            self.queue.extend(self.check_scripted_surveys())

