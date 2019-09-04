
__all__ = ['filter_swap_scheduler', 'simple_filter_sched']


class filter_swap_scheduler(object):
    """A simple way to schedule what filter to load
    """
    def __init__(self):
        pass

    def add_observation(self, observation):
        pass

    def __call__(self, conditions):
        """
        Returns
        -------
        list of strings for the filters that should be loaded
        """
        pass


class simple_filter_sched(filter_swap_scheduler):
    def __init__(self, illum_limit=10.):
        self.illum_limit = illum_limit

    def __call__(self, conditions):
        if conditions.moonPhase > self.illum_limit:
            result = ['g', 'r', 'i', 'z', 'y']
        else:
            result = ['u', 'g', 'r', 'i', 'y']
        return result
