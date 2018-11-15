from __future__ import absolute_import
from builtins import object
import numpy as np
import healpy as hp
from lsst.sims.utils import _hpid2RaDec
from lsst.sims.featureScheduler.utils import hp_in_lsst_fov, set_default_nside, hp_in_comcam_fov
import warnings
import logging


__all__ = ['Core_scheduler']


class Core_scheduler(object):
    """Core scheduler that takes completed obsrevations and observatory status and requests observations.
    """

    def __init__(self, surveys, nside=None, camera='LSST'):
        """
        Parameters
        ----------
        surveys : list (or list of lists) of lsst.sims.featureScheduler.survey objects
            A list of surveys to consider. If multiple surveys retrurn the same highest
            reward value, the survey at the earliest position in the list will be selected.
            Can also be a list of lists to make heirarchical priorities.
        nside : int
            A HEALpix nside value.
        camera : str ('LSST')
            Which camera to use for computing overlapping HEALpixels for an observation.
            Can be 'LSST' or 'comcam'
        """
        if nside is None:
            nside = set_default_nside()

        self.log = logging.getLogger("Core_scheduler")
        # initialize a queue of observations to request
        self.queue = []
        # Are the observations in the queue part of a sequence.
        self.queue_is_sequence = False
        # The indices of self.survey_lists that provided the last addition(s) to the queue
        self.survey_index = [None, None]

        # If we have a list of survey objects, convert to list-of-lists
        if isinstance(surveys[0], list):
            self.survey_lists = surveys
        else:
            self.survey_lists = [surveys]
        self.nside = nside
        hpid = np.arange(hp.nside2npix(nside))
        self.ra_grid_rad, self.dec_grid_rad = _hpid2RaDec(nside, hpid)
        # Should just make camera a class that takes a pointing and returns healpix indices
        if camera == 'LSST':
            self.pointing2hpindx = hp_in_lsst_fov(nside=nside)
        elif camera == 'comcam':
            self.pointing2hpindx = hp_in_comcam_fov(nside=nside)
        else:
            raise ValueError('camera %s not implamented' % camera)

    def flush_queue(self):
        """"
        Like it sounds, clear any currently queued desired observations.
        """
        self.queue = []
        self.queue_is_sequence = False
        self.survey_index = [None, None]

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
        indx = self.pointing2hpindx(observation['RA'], observation['dec'], observation['rotSkyPos'])
        for surveys in self.survey_lists:
            for survey in surveys:
                survey.add_observation(observation, indx=indx)

    def update_conditions(self, conditions_in):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position, loaded filters, cloud-mask, etc)
        """
        # Add the current queue and scheduled queue to the conditions
        self.conditions = conditions_in
        # put the local queue in the conditions
        self.conditions.queue = self.queue

        # XXX---TODO:  Could potentially put more complicated info from all
        # the surveys in the conditions object here. e.g., when a DDF plans to next request
        # observations.

    def request_observation(self):
        """
        Ask the scheduler what it wants to observe next

        Returns
        -------
        observation object (ra,dec,filter,rotangle)
        """
        if len(self.queue) == 0:
            self._fill_queue()

        if len(self.queue) == 0:
            return None
        else:
            # If the queue has gone stale, flush and refill. Zero means no flush_by was set.
            if (self.conditions.mjd > self.queue[0]['flush_by_mjd']) & (self.queue[0]['flush_by_mjd'] != 0):
                self.log.warning('Expired observations in queue, flushing and refilling')
                self.flush_queue()
                self._fill_queue()
            if len(self.queue) == 0:
                return None
            observation = self.queue.pop(0)
            if self.queue_is_sequence:
                if self.survey_lists[self.survey_index[0]][self.survey_index[1]].check_continue(observation, self.conditions):
                    return observation
                else:
                    self.log.warning('Sequence interrupted! Cleaning queue!')
                    self.flush_queue()
                    self._fill_queue()
                    if len(self.queue) == 0:
                        return None
                    else:
                        observation = self.queue.pop(0)
                        return observation
            else:
                return observation

    def _fill_queue(self):
        """
        Compute reward function for each survey and fill the observing queue with the
        observations from the highest reward survey.
        """

        rewards = None
        for ns, surveys in enumerate(self.survey_lists):
            rewards = np.zeros(len(surveys))
            for i, survey in enumerate(surveys):
                rewards[i] = np.nanmax(survey.calc_reward_function(self.conditions))
            # If we have a good reward, break out of the loop
            if np.nanmax(rewards) > -np.inf and np.nanmax(rewards) != hp.UNSEEN:
                self.survey_index[0] = ns
                break
        if np.all(np.bitwise_or(np.bitwise_or(np.isnan(rewards),
                                              np.isneginf(rewards)), rewards == hp.UNSEEN)):
            # All values are invalid
            self.flush_queue()
        else:
            try:
                to_fix = np.where(np.isnan(rewards))
                rewards[to_fix] = -np.inf
                # Take a min here, so the surveys will be executed in the order they are
                # entered if there is a tie.
                self.survey_index[1] = np.min(np.where(rewards == np.max(rewards)))

                # Survey return list of observations
                result = self.survey_lists[self.survey_index[0]][self.survey_index[1]](self.conditions)
                self.queue = result
                self.queue_is_sequence = self.survey_lists[self.survey_index[0]][self.survey_index[1]].sequence
            except ValueError as e:
                self.log.exception(e)
                self.flush_queue()
        if len(self.queue) == 0:
            self.log.warning('Failed to fill queue')
