import numpy as np
from utils import hp_in_lsst_fov


class Core_scheduler(object):
    """
    
    """

    def __init__(self, surveys, nside=32, camera='LSST'):
        """

        """
        # initialize a queue of observations to request
        self.queue = []
        self.surveys = surveys
        self.nside = nside
        if camera == 'LSST':
            self.pointing2hpindx = hp_in_lsst_fov(nside=nside)
        else:
            raise ValueError('')

    def flush_queue(self):
        """"
        Like it sounds, clear any currently queued desired observations.
        """
        self.queue = []

    def add_observation(self, observation):
        """
        Record a completed observation and update features accourdingly.

        Parameters
        ----------
        observation : dict-like
            An object that contains the relevant information about a 
            completed observation (e.g., mjd, ra, dec, filter, rotation angle, etc)
        """

        # XXX-- find the healpixel centers that are included in an observation
        indx = self.pointing2hpindx(observation['RA'], observation['Dec'])
        for survey in self.surveys:
            survey.add_observation(observation, indx=indx)

    def update_conditions(self, conditions):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position, loaded filters, cloud-mask, etc)
        """

        for survey in self.surveys:
            survey.update_conditions(conditions)

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
        
        """