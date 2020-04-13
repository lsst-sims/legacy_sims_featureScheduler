import numpy as np
from lsst.sims.featureScheduler.surveys import BaseSurvey
import copy
import lsst.sims.featureScheduler.basis_functions as basis_functions
from lsst.sims.featureScheduler.utils import empty_observation
from lsst.sims.featureScheduler import features
import logging
import random


__all__ = ['Deep_drilling_survey', 'generate_dd_surveys', 'dd_bfs']

log = logging.getLogger(__name__)


class Deep_drilling_survey(BaseSurvey):
    """A survey class for running deep drilling fields.

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
    flush_pad : float (30.)
        How long to hold observations in the queue after they were expected to be completed (minutes).
    """

    def __init__(self, basis_functions, RA, dec, sequence='rgizy',
                 nvis=[20, 10, 20, 26, 20],
                 exptime=30., nexp=2, ignore_obs=None, survey_name='DD',
                 reward_value=None, readtime=2., filter_change_time=120.,
                 nside=None, filter_match_shuffle=True, flush_pad=30., seed=42, detailers=None):
        super(Deep_drilling_survey, self).__init__(nside=nside, basis_functions=basis_functions,
                                                   detailers=detailers, ignore_obs=ignore_obs)
        random.seed(a=seed)

        self.ra = np.radians(RA)
        self.ra_hours = RA/360.*24.
        self.dec = np.radians(dec)
        self.survey_name = survey_name
        self.reward_value = reward_value
        self.flush_pad = flush_pad/60./24.  # To days
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
            self.filter_sequence = [obs['filter'] for obs in sequence]

        # Make an estimate of how long a seqeunce will take. Assumes no major rotational or spatial
        # dithering slowing things down.
        self.approx_time = np.sum([o['exptime']+readtime*o['nexp'] for o in self.observations])/3600./24. \
                           + filter_change_time*len(sequence)/3600./24.  # to days
        self.filter_match_shuffle = filter_match_shuffle
        self.filter_indices = {}
        self.filter_sequence = np.array(self.filter_sequence)
        for filtername in np.unique(self.filter_sequence):
            self.filter_indices[filtername] = np.where(self.filter_sequence == filtername)[0]

        if self.reward_value is None:
            self.extra_features['Ntot'] = features.N_obs_survey()
            self.extra_features['N_survey'] = features.N_obs_survey(note=self.survey_name)

    def check_continue(self, observation, conditions):
        # feasibility basis functions?
        '''
        This method enables external calls to check if a given observations that belongs to this survey is
        feasible or not. This is called once a sequence has started to make sure it can continue.

        XXX--TODO:  Need to decide if we want to develope check_continue, or instead hold the
        sequence in the survey, and be able to check it that way.
        '''

        result = True

        return result

    def calc_reward_function(self, conditions):
        result = -np.inf
        if self._check_feasibility(conditions):
            if self.reward_value is not None:
                result = self.reward_value
            else:
                # XXX This might backfire if we want to have DDFs with different fractions of the
                # survey time. Then might need to define a goal fraction, and have the reward be the
                # number of observations behind that target fraction.
                result = self.extra_features['Ntot'].feature / (self.extra_features['N_survey'].feature+1)
        return result

    def generate_observations_rough(self, conditions):
        result = []
        if self._check_feasibility(conditions):
            result = copy.deepcopy(self.observations)

            # Toss any filters that are not currently loaded
            result = [obs for obs in result if obs['filter'] in conditions.mounted_filters]

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


def dd_bfs(RA, dec, survey_name, ha_limits, frac_total=0.0185/2., aggressive_frac=0.011/2.):
    """
    Convienence function to generate all the feasibility basis functions
    """
    sun_alt_limit = -18.
    time_needed = 62.
    fractions = [0.00, aggressive_frac, frac_total]
    bfs = []
    bfs.append(basis_functions.Not_twilight_basis_function(sun_alt_limit=sun_alt_limit))
    bfs.append(basis_functions.Time_to_twilight_basis_function(time_needed=time_needed))
    bfs.append(basis_functions.Hour_Angle_limit_basis_function(RA=RA, ha_limits=ha_limits))
    bfs.append(basis_functions.Moon_down_basis_function())
    bfs.append(basis_functions.Fraction_of_obs_basis_function(frac_total=frac_total, survey_name=survey_name))
    bfs.append(basis_functions.Look_ahead_ddf_basis_function(frac_total, aggressive_frac,
                                                             sun_alt_limit=sun_alt_limit, time_needed=time_needed,
                                                             RA=RA, survey_name=survey_name,
                                                             ha_limits=ha_limits))
    bfs.append(basis_functions.Soft_delay_basis_function(fractions=fractions, delays=[0., 0.5, 1.5],
                                                         survey_name=survey_name))

    return bfs


def generate_dd_surveys(nside=None, nexp=2, detailers=None, reward_value=100,
                        frac_total=0.0185/2., aggressive_frac=0.011/2.):
    """Utility to return a list of standard deep drilling field surveys.

    XXX-Someone double check that I got the coordinates right!

    """

    surveys = []

    # ELAIS S1
    RA = 9.45
    dec = -44.
    survey_name = 'DD:ELAISS1'
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits, frac_total=frac_total, aggressive_frac=aggressive_frac)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='urgizy',
                                        nvis=[8, 20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=reward_value,
                                        nside=nside, nexp=nexp, detailers=detailers))

    # XMM-LSS
    survey_name = 'DD:XMM-LSS'
    RA = 35.708333
    dec = -4-45/60.
    ha_limits = ([0., 1.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits, frac_total=frac_total, aggressive_frac=aggressive_frac)

    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='urgizy',
                                        nvis=[8, 20, 10, 20, 26, 20], survey_name=survey_name, reward_value=reward_value,
                                        nside=nside, nexp=nexp, detailers=detailers))

    # Extended Chandra Deep Field South
    RA = 53.125
    dec = -28.-6/60.
    survey_name = 'DD:ECDFS'
    ha_limits = [[0.5, 3.0], [20., 22.5]]
    bfs = dd_bfs(RA, dec, survey_name, ha_limits, frac_total=frac_total, aggressive_frac=aggressive_frac)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='urgizy',
                                        nvis=[8, 20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=reward_value, nside=nside,
                                        nexp=nexp, detailers=detailers))

    # COSMOS
    RA = 150.1
    dec = 2.+10./60.+55/3600.
    survey_name = 'DD:COSMOS'
    ha_limits = ([0., 2.5], [21.5, 24.])
    bfs = dd_bfs(RA, dec, survey_name, ha_limits, frac_total=frac_total, aggressive_frac=aggressive_frac)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence='urgizy',
                                        nvis=[8, 20, 10, 20, 26, 20],
                                        survey_name=survey_name, reward_value=reward_value, nside=nside,
                                        nexp=nexp, detailers=detailers))

    # Euclid Fields
    # I can use the sequence kwarg to do two positions per sequence
    filters = 'urgizy'
    nviss = [8, 5, 7, 19, 24, 5]
    survey_name = 'DD:EDFS'
    # Note the sequences need to be in radians since they are using observation objects directly
    RAs = np.radians([58.97, 63.6])
    decs = np.radians([-49.28, -47.60])
    sequence = []
    exptime = 30
    for filtername, nvis in zip(filters, nviss):
        for ra, dec in zip(RAs, decs):
            for num in range(nvis):
                obs = empty_observation()
                obs['filter'] = filtername
                obs['exptime'] = exptime
                obs['RA'] = ra
                obs['dec'] = dec
                obs['nexp'] = nexp
                obs['note'] = survey_name
                sequence.append(obs)

    ha_limits = ([0., 1.5], [22.5, 24.])
    # And back to degrees for the basis function
    bfs = dd_bfs(np.degrees(RAs[0]), np.degrees(decs[0]), survey_name, ha_limits,
                 frac_total=frac_total, aggressive_frac=aggressive_frac)
    surveys.append(Deep_drilling_survey(bfs, RA, dec, sequence=sequence,
                                        survey_name=survey_name, reward_value=reward_value, nside=nside,
                                        nexp=nexp, detailers=detailers))

    return surveys
