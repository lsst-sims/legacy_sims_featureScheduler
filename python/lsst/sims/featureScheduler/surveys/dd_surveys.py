import numpy as np
from lsst.sims.featureSceduler.surveys import BaseSurvey

class Deep_drilling_survey(BaseSurvey):
    """A survey class for running deep drilling fields
    """
    # XXX--maybe should switch back to taking basis functions and weights to
    # make it easier to put in masks for moon and limits for seeing?
    def __init__(self, RA, dec, extra_features=None, sequence='rgizy',
                 nvis=[20, 10, 20, 26, 20],
                 exptime=30.,
                 nexp=2, ignore_obs='dummy', survey_name='DD', fraction_limit=0.01,
                 ha_limits=([0., 1.5], [21.0, 24.]), reward_value=101., moon_up=True, readtime=2.,
                 avoid_same_day=False,
                 day_space=2., max_clouds=0.7, moon_distance=30., filter_goals=None, nside=None):
        """
        Parameters
        ----------
        RA : float
            The RA of the field (degrees)
        dec : float
            The dec of the field to observe (degrees)
        extra_features : list of feature objects (None)
            The features to track, will construct automatically if None.
        sequence : list of observation objects or str (rgizy)
            The sequence of observations to take. Can be a string of list of obs objects.
        nvis : list of ints
            The number of visits in each filter. Should be same length as sequence.
        survey_name : str (DD)
            The name to give this survey so it can be tracked
        fraction_limit : float (0.01)
            Do not request observations if the fraction of observations from this
            survey exceeds the frac_limit.
        ha_limits : list of floats ([-1.5, 1.])
            The range of acceptable hour angles to start a sequence (hours)
        reward_value : float (101.)
            The reward value to report if it is able to start (unitless).
        moon_up : bool (True)
            Require the moon to be up (True) or down (False) or either (None).
        readtime : float (2.)
            Readout time for computing approximate time of observing the sequence. (seconds)
        day_space : float (2.)
            Demand this much spacing between trying to launch a sequence (days)
        max_clouds : float (0.7)
            Maximum allowed cloud value for an observation.
        """
        # No basis functions for this survey
        basis_functions = []
        basis_weights = []
        self.ra = np.radians(RA)
        self.ra_hours = RA/360.*24.
        self.dec = np.radians(dec)
        self.ignore_obs = ignore_obs
        self.survey_name = survey_name
        self.HA_limits = np.array(ha_limits)
        self.reward_value = reward_value
        self.moon_up = moon_up
        self.fraction_limit = fraction_limit
        self.day_space = day_space
        self.survey_id = 5
        self.nside = nside
        self.filter_list = []
        self.max_clouds = max_clouds
        self.moon_distance = np.radians(moon_distance)
        self.sequence = True  # Specifies the survey gives sequence of observations
        self.avoid_same_day = avoid_same_day
        self.filter_goals = filter_goals

        if extra_features is None:
            self.extra_features = {}
            # Current filter
            self.extra_features['current_filter'] = features.Current_filter()
            # Available filters
            self.extra_features['mounted_filters'] = features.Mounted_filters()
            # Observatory information
            self.extra_features['observatory'] = features.Observatory({'readtime': readtime,
                                                                       'filter_change_time': 120.}
                                                                      )  # FIXME:
            self.extra_features['night'] = features.Current_night()
            # The total number of observations
            self.extra_features['N_obs'] = features.N_obs_count()
            # The number of observations for this survey
            self.extra_features['N_obs_self'] = features.N_obs_survey(note=survey_name)
            # The current LMST. Pretty sure in hours
            self.extra_features['lmst'] = features.Current_lmst()
            # Moon altitude
            self.extra_features['sun_moon_alt'] = features.Sun_moon_alts()
            # Moon altitude
            self.extra_features['moon'] = features.Moon()

            # Time to next moon rise

            # Time to twilight

            # last time this survey was observed (in case we want to force a cadence)
            self.extra_features['last_obs_self'] = features.Last_observation(survey_name=self.survey_name)
            # last time a sequence observation
            self.extra_features['last_seq_obs'] = features.LastSequence_observation(sequence_ids=[self.survey_id])

            # Current MJD
            self.extra_features['mjd'] = features.Current_mjd()
            # Observable time. This includes altitude and night limits
            # Fixme: add proper altitude limit from ha limits
            self.extra_features['night_boundaries'] = features.CurrentNightBoundaries()
            # Proposal information
            self.extra_features['proposals'] = features.SurveyProposals(ids=(self.survey_id,),
                                                                        names=(self.survey_name,))
            # Cloud cover information
            self.extra_features['bulk_cloud'] = features.BulkCloudCover()
        else:
            self.extra_features = extra_features

        super(Deep_drilling_survey, self).__init__(basis_functions=basis_functions,
                                                   basis_weights=basis_weights,
                                                   extra_features=self.extra_features,
                                                   nside=nside)

        if type(sequence) == str:
            opsim_fields = read_fields()
            self.pointing2hpindx = hp_in_lsst_fov(nside=self.nside)
            hp2fields = np.zeros(hp.nside2npix(self.nside), dtype=np.int)
            for i in range(len(opsim_fields['RA'])):
                hpindx = self.pointing2hpindx(opsim_fields['RA'][i], opsim_fields['dec'][i])
                hp2fields[hpindx] = i+1
            hpid = _raDec2Hpid(self.nside, self.ra, self.dec)

            fields = read_fields()
            field = fields[hp2fields[hpid]]
            field['tag'] = self.survey_id
            self.fields = [field]

            self.sequence = []
            self.sequence_dict = dict()
            filter_list = []
            for num, filtername in zip(nvis, sequence):
                filter_list.append(filtername)
                if filtername not in self.sequence_dict:
                    self.sequence_dict[filtername] = []

                for j in range(num):
                    obs = empty_observation()
                    obs['filter'] = filtername
                    obs['exptime'] = exptime
                    obs['RA'] = self.ra
                    obs['dec'] = self.dec
                    obs['nexp'] = nexp
                    obs['note'] = survey_name
                    obs['field_id'] = hp2fields[hpid]
                    obs['survey_id'] = self.survey_id

                    # self.sequence.append(obs)
                    self.sequence_dict[filtername].append(obs)
            self.filter_list = np.unique(np.array(filter_list))
        else:
            self.sequence_dict = None
            self.sequence = sequence

        # add extra features to map filter goals
        for filtername in self.filter_list:
            self.extra_features['N_obs_%s' % filtername] = features.N_obs_count(filtername=filtername)

        self.approx_time = np.sum([(o['exptime']+readtime)*o['nexp'] for o in obs])

        # Construct list of all the filters that need to be loaded to execute sequence
        self.filter_set = set(self.filter_list)

    def _check_feasability(self):
        # Check that all filters are available
        result = self.filter_set.issubset(set(self.extra_features['mounted_filters'].feature))
        if not result:
            return False

        if (self.avoid_same_day and
                (self.extra_features['last_seq_obs'].feature['night'] == self.extra_features['night'].feature)):
            return False
        # Check if the LMST is in range
        HA = self.extra_features['lmst'].feature - self.ra_hours
        HA = HA % 24.

        result = False
        for limit in self.HA_limits:
            lres = limit[0] <= HA < limit[1]
            result = result or lres

        if not result:
            return False
        # Check moon alt
        if self.moon_up is not None:
            if self.moon_up:
                if self.extra_features['sun_moon_alt'].feature['moonAlt'] < 0.:
                    return False
            else:
                if self.extra_features['sun_moon_alt'].feature['moonAlt'] > 0.:
                    return False

        # Make sure twilight hasn't started
        if self.extra_features['sun_moon_alt'].feature['sunAlt'] > np.radians(-18.):
            return False

        # Check that it's been long enough since last sequence
        if self.extra_features['mjd'].feature - self.extra_features['last_obs_self'].feature['mjd'] < self.day_space:
            return False

        # TODO: Check if the moon will come up. Compare next moonrise time to self.apporox time

        # TODO: Check if twilight starts soon

        # TODO: Make sure it is possible to complete the sequence of observations. Hit any limit?

        # Check if there's still enough time to complete the observation
        time_left = (self.extra_features['night_boundaries'].feature['next_twilight_start'] -
                     self.extra_features['mjd'].feature) * 24.*60.*60.  # convert to seconds

        seq_time = 42.  # Make sure there is enough time for an extra visit after the DD sequence
        current_filter = self.extra_features['current_filter'].feature
        for obs in self.sequence:
            for o in obs:
                if current_filter != o['filter']:
                    seq_time += self.extra_features['observatory'].feature['filter_change_time']
                    current_filter = o['filter']
                seq_time += o['exptime']+self.extra_features['observatory'].feature['readtime']*o['nexp']

        # log.debug('Time left: %.2f | Approx. time: %.2f' % (time_left, seq_time))
        if time_left < seq_time:
            return False

        if self.extra_features['N_obs'].feature == 0:
            return True

        # Check if we are over-observed relative to the fraction of time alloted.
        if self.extra_features['N_obs_self'].feature/float(self.extra_features['N_obs'].feature) > self.fraction_limit:
            return False

        # Check clouds
        if self.extra_features['bulk_cloud'].feature > self.max_clouds:
            return False

        # If we made it this far, good to go
        return result

    def check_feasibility(self, observation):
        '''
        This method enables external calls to check if a given observations that belongs to this survey is
        feasible or not. This is called once a sequence has started to make sure it can continue.

        :return:
        '''

        # Check moon distance
        if self.moon_up is not None:
            moon_separation = _angularSeparation(self.extra_features['moon'].feature['moonRA'],
                                                 self.extra_features['moon'].feature['moonDec'],
                                                 observation['RA'],
                                                 observation['dec'])
            if moon_separation < self.moon_distance:
                return False

        # Check clouds
        if self.extra_features['bulk_cloud'].feature > self.max_clouds:
            return False

        # If we made it this far, good to go
        return True

    def get_sequence(self):
        '''
        Build and return sequence of observations
        :return:
        '''

        if self.sequence_dict is None:
            return self.sequence
        elif len(self.filter_list) == 1:
            return self.sequence_dict[self.filter_list[0]]
        elif self.filter_goals is None:
            self.sequence = []
            for filtername in self.filter_list:
                for observation in self.sequence_dict[filtername]:
                    self.sequence.append(observation)
            return self.sequence

        # If arrived here, then need to construct sequence. Will but the current filter first and the one that
        # requires more observations last
        filter_need = np.zeros(len(self.filter_list))
        filter_goal = np.array([self.filter_goals[fname] for fname in self.filter_list])
        filter_goal = (1.-filter_goal)/(1.+filter_goal)

        if self.extra_features['N_obs'].feature > 0:
            for i, filtername in enumerate(self.filter_list):
                filter_need[i] = ((self.extra_features['N_obs'].feature -
                                   self.extra_features['N_obs_%s' % filtername].feature) /
                                  (self.extra_features['N_obs'].feature +
                                   self.extra_features['N_obs_%s' % filtername].feature)) / filter_goal[i]
        else:
            filter_need = 1./filter_goal

        filter_order = np.array(self.filter_list[np.argsort(filter_need)[::-1]])
        current_filter_index = np.where(filter_order == self.extra_features['current_filter'].feature)[0]
        if current_filter_index != 0 and current_filter_index != len(self.filter_list)-1:
            first_filter = filter_order[0]
            filter_order[0] = self.extra_features['current_filter'].feature
            filter_order[current_filter_index] = first_filter
        elif current_filter_index == len(self.filter_list)-1:
            filter_order = np.append(filter_order[-1], filter_order[:-1])

        log.debug('DeepDrilling[filter_order]: %s was %s[need: %s] ' % (filter_order,
                                                                        self.filter_list,
                                                                        filter_need))
        self.sequence = []
        for filtername in filter_order:
            for observation in self.sequence_dict[filtername]:
                self.sequence.append(observation)
        return self.sequence


    def calc_reward_function(self):
        result = -np.inf
        if self._check_feasability():
            result = self.reward_value
        return result

    def __call__(self):
        result = []
        if self._check_feasability():
            result = copy.deepcopy(self.get_sequence())
            # Note, could check here what the current filter is and re-order the result
        return result


def generate_dd_surveys(nside=None):
    """Utility to return a list of standard deep drilling field surveys.

    XXX-Someone double check that I got the coordinates right!

    """

    surveys = []
    # ELAIS S1
    surveys.append(Deep_drilling_survey(9.45, -44., sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name='DD:ELAISS1', reward_value=100, moon_up=None,
                                        fraction_limit=0.0185, ha_limits=([0., 1.18], [21.82, 24.]),
                                        nside=nside))
    surveys.append(Deep_drilling_survey(9.45, -44., sequence='u',
                                        nvis=[7],
                                        survey_name='DD:u,ELAISS1', reward_value=100, moon_up=False,
                                        fraction_limit=0.0015, ha_limits=([0., 1.18], [21.82, 24.]),
                                        nside=nside))

    # XMM-LSS
    surveys.append(Deep_drilling_survey(35.708333, -4-45/60., sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name='DD:XMM-LSS', reward_value=100, moon_up=None,
                                        fraction_limit=0.0185, ha_limits=([0., 1.3], [21.7, 24.]),
                                        nside=nside))
    surveys.append(Deep_drilling_survey(35.708333, -4-45/60., sequence='u',
                                        nvis=[7],
                                        survey_name='DD:u,XMM-LSS', reward_value=100, moon_up=False,
                                        fraction_limit=0.0015, ha_limits=([0., 1.3], [21.7, 24.]),
                                        nside=nside))

    # Extended Chandra Deep Field South
    surveys.append(Deep_drilling_survey(53.125, -28.-6/60., sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name='DD:ECDFS', reward_value=100, moon_up=None,
                                        fraction_limit=0.0185, ha_limits=[[0.5, 3.0], [20., 22.5]],
                                        nside=nside))
    surveys.append(Deep_drilling_survey(53.125, -28.-6/60., sequence='u',
                                        nvis=[7],
                                        survey_name='DD:u,ECDFS', reward_value=100, moon_up=False,
                                        fraction_limit=0.0015, ha_limits=[[0.5, 3.0], [20., 22.5]],
                                        nside=nside))
    # COSMOS
    surveys.append(Deep_drilling_survey(150.1, 2.+10./60.+55/3600., sequence='rgizy',
                                        nvis=[20, 10, 20, 26, 20],
                                        survey_name='DD:COSMOS', reward_value=100, moon_up=None,
                                        fraction_limit=0.0185, ha_limits=([0., 1.5], [21.5, 24.]),
                                        nside=nside))
    surveys.append(Deep_drilling_survey(150.1, 2.+10./60.+55/3600., sequence='u',
                                        nvis=[7], ha_limits=([0., 1.5], [21.5, 24.]),
                                        survey_name='DD:u,COSMOS', reward_value=100, moon_up=False,
                                        fraction_limit=0.0015,
                                        nside=nside))

    return surveys
