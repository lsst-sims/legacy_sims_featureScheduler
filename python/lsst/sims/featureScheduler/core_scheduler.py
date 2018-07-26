from __future__ import absolute_import
from builtins import object
import numpy as np
import healpy as hp
from lsst.sims.utils import _hpid2RaDec
from .utils import hp_in_lsst_fov, set_default_nside, hp_in_comcam_fov
import warnings
import logging

default_nside = None


class Core_scheduler(object):
    """Core scheduler that takes completed obsrevations and observatory status and requests observations.
    """

    def __init__(self, surveys, nside=default_nside, camera='LSST'):
        """
        Parameters
        ----------
        surveys : list of survey objects
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
        self.is_sequence = False
        self.survey_index = [0, 0]

        # If we have a list of survey objects, convert to list-of-lists
        if isinstance(surveys[0], list):
            self.survey_lists = surveys
        else:
            self.survey_lists = [surveys]
        self.nside = nside
        hpid = np.arange(hp.nside2npix(nside))
        self.ra_grid_rad, self.dec_grid_rad = _hpid2RaDec(nside, hpid)
        self.conditions = None
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
        # XXX-in the future, we may want to refactor to support multiple nside resolutions
        # I think indx would then be a dict with keys 32,64,128, etc. Then each feature would
        # say indx = indx[self.nside]
        indx = self.pointing2hpindx(observation['RA'], observation['dec'], observation['rotSkyPos'])
        for surveys in self.survey_lists:
            for survey in surveys:
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
        for surveys in self.survey_lists:
            for survey in surveys:
                survey.update_conditions(conditions)
        self.conditions = conditions

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
            self.log.warning('Failed to fill queue')
            # self._fill_queue()
            return None
        else:
            observation = self.queue.pop(0)
            if self.is_sequence:
                if self.survey_lists[self.survey_index[0]][self.survey_index[1]].check_feasibility(observation):
                    return observation
                else:
                    self.log.warning('Sequence interrupted! Cleaning queue!')
                    self._clean_queue()
                    return None
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
                rewards[i] = np.max(survey.calc_reward_function())
            # If we have a good reward, break out of the loop
            if np.nanmax(rewards) > -np.inf:
                self.survey_index[0] = ns
                break

        # Take a min here, so the surveys will be executed in the order they are
        # entered if there is a tie.
        if np.all(np.bitwise_or(np.isnan(rewards), np.isneginf(rewards))):
            # All values are invalid
            self._clean_queue()
        else:
            try:
                to_fix = np.where(np.isnan(rewards))
                rewards[to_fix] = -np.inf
                self.survey_index[1] = np.min(np.where(rewards == np.max(rewards)))

                # Survey return list of observations
                result = self.survey_lists[self.survey_index[0]][self.survey_index[1]]()
                self.queue = result
                self.is_sequence = self.survey_lists[self.survey_index[0]][self.survey_index[1]].sequence
            except ValueError as e:
                self.log.exception(e)
                self._clean_queue()

    def _clean_queue(self):
        """
        Clean queue.
        """
        self.queue = []
        self.is_sequence = False
        self.survey_index = None

class Core_scheduler_parallel(Core_scheduler):
    """Execute survey methods in parallel
    """
    def __init__(self, surveys, nside=default_nside, camera='LSST'):
        """
        Before running, start ipyparallel engines at the command line with something like:
        > ipcluster start -n 7
        where the final number is the number of surveys you will be running
        """
        super(Core_scheduler_parallel, self).__init__(surveys, nside=nside, camera=camera)
        # Hide import here in case ipyparallel is not part of standard install
        import ipyparallel as ipp
        # Set up the connection to the ipython engines
        self.rc = ipp.Client()
        self.dview = self.rc[:]
        # We always want blocking execution
        self.dview.block=False
        # Check that we have enough engines for surveys
        if len(self.surveys) > len(self.rc):
            raise ValueError('Not enough ipcluster engines. Trying to run %i surveys on %i engines' % (len(self.surveys), len(self.rc)))

        # Make sure the engines have numpy. Note, "as np" is ignored on engines.
        with self.dview.sync_imports():
            import numpy as np

        # Put one survey on each engine
        for i, survey in enumerate(surveys):
            self.rc[i].push({'survey': survey})
    #XXX--need a method to pull all the survey objects back into self.surveys

    def add_observation(self, observation):
        indx = self.pointing2hpindx(observation['RA'], observation['dec'])
        self.dview.push({'indx': indx})
        self.dview.push({'observation': observation})
        result = self.dview.execute('survey.add_observation(observation, indx=indx)')

    def update_conditions(self, conditions):
        # Add the current queue and scheduled queue to the conditions
        conditions['queue'] = self.queue
        self.dview.push({'conditions': conditions})
        result = self.dview.execute('survey.update_conditions(conditions)')
        self.conditions = conditions

    def _fill_queue(self):
        """
        Compute reward function for each survey and fill the observing queue with the
        observations from the highest reward survey.
        """
        self.dview.execute('reward = numpy.max(survey.calc_reward_function())')
        rewards = self.dview['reward']
        # Take a min here, so the surveys will be executed in the order they are
        # entered if there is a tie.
        good = int(np.min(np.where(rewards == np.max(rewards))))
        # Survey return list of observations
        result = self.rc[good].execute('result = survey()')
        result = self.rc[good]['result']
        self.queue = result

