import numpy as np
from lsst.sims.featureScheduler.surveys import BaseSurvey
import copy
import lsst.sims.featureScheduler.basis_functions as basis_functions
from lsst.sims.featureScheduler.utils import empty_observation
import healpy as hp
import random



class DESC_ddf(BaseSurvey):
    """DDF survey based on Scolnic et al Cadence White Paper.
    """
    def __init__(self, basis_functions, RA, dec, sequences=None,
                 exptime=30., nexp=1, ignore_obs=None, survey_name='DD_DESC',
                 reward_value=101., readtime=2., filter_change_time=120.,
                 nside=None, flush_pad=30., seed=42, detailers=None):
        super(DESC_ddf, self).__init__(nside=nside, basis_functions=basis_functions,
                                       detailers=detailers, ignore_obs=ignore_obs)

        self.ra = np.radians(RA)
        self.ra_hours = RA/360.*24.
        self.dec = np.radians(dec)
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad/60./24.  # To days
        self.sequence = True  # Specifies the survey gives sequence of observations

        self.simple_obs = empty_observation
        self.simple_obs['RA'] = np.radians(RA)
        self.simple_obs['dec'] = np.radians(dec)
        self.simple_obs['exptime'] = exptime
        self.simple_obs['nexp'] = nexp
        self.simple_obs['note'] = survey_name

        # Define the sequences we would like to do
        if sequences is None:
            self.sequences = [{'g': 2, 'r': 4, 'i': 8}, {'z': 25, 'y': 4}, None]
        else:
            self.sequences = sequences

        # Track what we last tried to do
        # XXX-this should probably go into self.extra_features or something for consistency.
        self.sequence_index = 0
        self.last_night_observed = -100

    def _check_feasibility(self, conditions):
        """
        Check if the survey is feasable in the current conditions
        """
        # Advance the sequence index if we have skipped a day intentionally
        if (self.sequences[self.sequence_index] is None) & (conditions['night']-self.last_night_observed > 1):
            self.sequence_index = (self.sequence_index + 1) % (len(self.sequences) + 1)

        if self.sequences[self.sequence_index] is None:
            return False

        for bf in self.basis_functions:
            result = bf.check_feasibility(conditions)
            if not result:
                return result
        return result

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            result = self.reward_value
        return result

    def generate_observations_rough(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            for key in self.sequences[self.sequence_index]:
                if [key] in conditions['mounted_filters']:
                    for i in range(self.sequences[key]):
                        temp_obs = self.simple_obs.copy()
                        temp_obs['filter'] = key
                        result.append(temp_obs)
            # Just assuming this sequence gets observed.
            self.last_night_observed = conditions['night']
            self.sequence_index = (self.sequence_index + 1) % (len(self.sequences) + 1)
        return result


