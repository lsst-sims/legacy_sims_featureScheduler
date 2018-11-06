import numpy as np
from lsst.sims.featureScheduler.surveys import BaseSurvey
import copy
import lsst.sims.featureScheduler.basis_functions as basis_functions
from lsst.sims.featureScheduler.utils import empty_observation
from lsst.sims.utils import _angularSeparation
import logging
import healpy as hp
import random


__all__ = ['Deep_drilling_survey', 'generate_dd_surveys']

log = logging.getLogger(__name__)


class Deep_drilling_survey(BaseSurvey):
    """A survey class for running deep drilling fields
    """

    def __init__(self, basis_functions, RA, dec, sequence='rgizy',
                 nvis=[20, 10, 20, 26, 20],
                 exptime=30., nexp=2, ignore_obs='dummy', survey_name='DD',
                 reward_value=101., readtime=2., filter_change_time=120.,
                 nside=None, filter_match_shuffle=True, flush_pad=30., seed=42):
        """
        Parameters
        ----------
        basis_functions : list of lsst.sims.featureScheduler.basis_function objects
            These should be feasibility basis functions.
        RA : float
            The RA of the field (degrees)
        dec : float
            The dec of the field to observe (degrees)
        sequence : list of observation objects or str (rgizy)
            The sequence of observations to take. Can be a string of list of obs objects.
        nvis : list of ints
            The number of visits in each filter. Should be same length as sequence.
        survey_name : str (DD)
            The name to give this survey so it can be tracked
        reward_value : float (101.)
            The reward value to report if it is able to start (unitless).
        readtime : float (2.)
            Readout time for computing approximate time of observing the sequence. (seconds)
        filter_match_shuffle : bool (True)
            If True, switch up the order filters are executed in (first sequence will be currently
            loaded filter if possible)
        flush_pad : float (10.)
            How long to hold observations in the queue after they were expected to be completed (minutes).
        """
        super(Deep_drilling_survey, self).__init__(nside=nside, basis_functions=basis_functions)
        random.seed(a=seed)

        self.ra = np.radians(RA)
        self.ra_hours = RA/360.*24.
        self.dec = np.radians(dec)
        self.ignore_obs = ignore_obs
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad/60./24.  # To days
        self.sequence = True  # Specifies the survey gives sequence of observations
        self.filter_sequence = []
        if type(sequence) == str:
            self.observations = []
            for num, filtername in zip(nvis, sequence):
                for j in range(num):
                    obs = empty_observation()
                    obs['filter'] = filtername
                    obs['exptime'] = exptime
                    obs['RA'] = self.ra
                    obs['dec'] = self.dec
                    obs['nexp'] = nexp
                    obs['note'] = survey_name
                    self.observations.append(obs)
                    self.filter_sequence.append(filtername)
        else:
            self.observations = sequence

        # Make an estimate of how long a seqeunce will take. Assumes no major rotational or spatial
        # dithering slowing things down.
        self.approx_time = np.sum([o['exptime']+readtime*o['nexp'] for o in self.observations])/3600./24. \
                           + filter_change_time*len(sequence)/3600./24.  # to days
        self.filter_match_shuffle = filter_match_shuffle
        self.filter_indices = {}
        self.filter_sequence = np.array(self.filter_sequence)
        for filtername in np.unique(self.filter_sequence):
            self.filter_indices[filtername] = np.where(self.filter_sequence == filtername)[0]

    def check_continue(self, observation, conditions):
        # feasibility basis functions?
        '''
        This method enables external calls to check if a given observations that belongs to this survey is
        feasible or not. This is called once a sequence has started to make sure it can continue.

        :return:
        '''

        result = True
        #for bf in self.basis_functions:
        #    result = bf.check_feasibility(conditions)
        #    if not result:
        #        return result

        return result

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            result = self.reward_value
        return result

    def __call__(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            result = copy.deepcopy(self.observations)

            if self.filter_match_shuffle:
                filters_remaining = list(self.filter_indices.keys())
                random.shuffle(filters_remaining)
                # If we want to observe the currrent filter, put it first
                if conditions.current_filter in filters_remaining:
                    filters_remaining.insert(0, filters_remaining.pop(filters_remaining.index(conditions.current_filter)))
                final_result = []
                for filtername in filters_remaining:
                    final_result.extend(result[np.min(self.filter_indices[filtername]):np.max(self.filter_indices[filtername])+1])
                result = final_result
            # Let's set the mjd to flush the queue by
            for i, obs in enumerate(result):
                result[i]['flush_by_mjd'] = conditions.mjd + self.approx_time + self.flush_pad
        return result


def dd_bfs(RA, dec, survey_name, ha_limits, frac_total=0.0185):
    """
    Convienence function to generate all the feasibility basis functions
    """
    bfs = []
    bfs.append(basis_functions.Filter_loaded_basis_function(filternames=['r', 'g', 'i', 'z', 'y']))
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=-18.5))  # XXX-possible pyephem bug
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=62.))
    bfs.append(basis_functions.Force_delay_basis_function(days_delay=2., survey_name=survey_name))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.Fraction_of_obs_basis_function(frac_total=frac_total, survey_name=survey_name))
    bfs.append(basis_functions.Clouded_out_basis_function())

    return bfs


def dd_u_bfs(RA, dec, survey_name, ha_limits):
    """Convienence function to generate all the feasibility basis functions for u-band DDFs
    """
    bfs = []
    bfs.append(basis_functions.Filter_loaded_basis_function(filternames='u'))
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=-18.5))  # XXX-possible pyephem bug
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=6.))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))

    bfs.append(basis_functions.Force_delay_basis_function(days_delay=2., survey_name=survey_name))
    bfs.append(basis_functions.Moon_down_basis_function())
    bfs.append(basis_functions.Fraction_of_obs_basis_function(frac_total=0.0015, survey_name=survey_name))
    bfs.append(basis_functions.Clouded_out_basis_function())

    return bfs


def generate_dd_surveys(nside=None):
    """Utility to return a list of standard deep drilling field surveys.

    XXX-Someone double check that I got the coordinates right!

    """

    surveys = []

    # ELAIS S1
    RA = 9.45
    dec = -44.
    survey_name = 'DD:ELAISS1'
    ha_limits = ([0., 1.18], [21.82, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=100,
                                        nside=nside))

    survey_name = 'DD:u,ELAISS1'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[7], survey_name=survey_name, reward_value=100, nside=nside))

    # XMM-LSS
    survey_name = 'DD:XMM-LSS'
    RA = 35.708333
    dec = -4-45/60.
    ha_limits = ([0., 1.3], [21.7, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20], survey_name=survey_name, reward_value=100,
                                        nside=nside))
    survey_name = 'DD:u,XMM-LSS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[7], survey_name=survey_name, reward_value=100, nside=nside))

    # Extended Chandra Deep Field South
    RA = 53.125
    dec = -28.-6/60.
    survey_name = 'DD:ECDFS'
    ha_limits = [[0.5, 3.0], [20., 22.5]]
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=100, nside=nside))

    survey_name = 'DD:u,ECDFS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[7], survey_name=survey_name, reward_value=100, nside=nside))
    # COSMOS
    RA = 150.1
    dec = 2.+10./60.+55/3600.
    survey_name = 'DD:COSMOS'
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=100, nside=nside))
    survey_name = 'DD:u,COSMOS'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u',
                                        nvis=[7], survey_name=survey_name, reward_value=100, nside=nside))

    # Extra DD Field, just to get to 5. Still not closed on this one
    survey_name = 'DD:290'
    RA = 349.386443
    dec = -63.321004
    ha_limits = ([0., 0.5], [23.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=100, nside=nside))

    survey_name = 'DD:u,290'
    bfs = dd_u_bfs(RA, dec, survey_name, ha_limits)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='u', nvis=[7],
                                        survey_name=survey_name, reward_value=100, nside=nside))

    return surveys
