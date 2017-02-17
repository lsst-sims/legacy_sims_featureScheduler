import numpy as np


###  XXX
# Maybe we can define a Survey class--each survey can then contain basis functions and features.
# The Core_scheduler then calls each Survey and picks the one with the best cost function. 


class Core_scheduler(object):
    """
    
    """

    def __init__(self):
        """

        """
        # initialize a queue of observations to request
        self.queue = []

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

        pass

    def update_conditions(self, conditions):
        """
        Parameters
        ----------
        conditions : dict-like
            The current conditions of the telescope (pointing position, loaded filters, cloud-mask, etc)
        """

        pass

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