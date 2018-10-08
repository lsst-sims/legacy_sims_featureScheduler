class Greedy_survey_fields(BaseSurvey):
    """
    Use a field tessellation and assign each healpix to a field.
    """
    def __init__(self, basis_functions, basis_weights, extra_features=None, filtername='r',
                 block_size=25, smoothing_kernel=None, nside=default_nside,
                 dither=False, seed=42, ignore_obs='ack', survey_name='',
                 tag_fields=False, tag_map=None, tag_names=None, extra_basis_functions=None):
        if extra_features is None:
            extra_features = {}
            extra_features['night'] = features.Current_night()
            extra_features['mounted_filters'] = features.Mounted_filters()
        if tag_fields and tag_names is not None:
            extra_features['proposals'] = features.SurveyProposals(ids=tag_names.keys(),
                                                                   names=tag_names.values())
        super(Greedy_survey_fields, self).__init__(basis_functions=basis_functions,
                                                   basis_weights=basis_weights,
                                                   extra_features=extra_features,
                                                   smoothing_kernel=smoothing_kernel,
                                                   ignore_obs=ignore_obs,
                                                   nside=nside,
                                                   extra_basis_functions=extra_basis_functions,
                                                   survey_name=survey_name)
        self.filtername = filtername
        # Load the OpSim field tessellation
        self.fields_init = read_fields()
        self.fields = self.fields_init.copy()
        self.block_size = block_size
        np.random.seed(seed)
        self.dither = dither
        self.night = extra_features['night'].feature + 0
        for bf in self.basis_functions:
            if 'hp2fields' in bf.condition_features:
                bf.condition_features['hp2fields'].update_conditions({'hp2fields':self.hp2fields})

        self.tag_map = tag_map
        self.tag_fields = tag_fields
        # self.inside_tagged = np.zeros_like(self.hp2fields) == 0

        if tag_fields:
            tags = np.unique(tag_map[tag_map > 0])
            for tag in tags:
                inside_tag = np.where(tag_map == tag)
                fields_id = np.unique(self.hp2fields[inside_tag])
                self.fields['tag'][fields_id] = tag
        else:
            for i in range(len(self.fields)):
                self.fields['tag'][i] = 1

    def _check_feasability(self):
        """
        Check if the survey is feasible in the current conditions
        """
        feasibility = self.filtername in self.extra_features['mounted_filters'].feature
        # return feasibility
        for bf in self.basis_functions:
            # log.debug('Check feasability: [%s] %s %s' % (str(bf), feasibility, bf.check_feasibility()))
            feasibility = feasibility and bf.check_feasibility()
            if not feasibility:
                break

        return feasibility

    def _spin_fields(self, lon=None, lat=None, lon2=None):
        """Spin the field tessellation to generate a random orientation

        The default field tesselation is rotated randomly in longitude, and then the
        pole is rotated to a random point on the sphere.

        Parameters
        ----------
        lon : float (None)
            The amount to initially rotate in longitude (radians). Will use a random value
            between 0 and 2 pi if None (default).
        lat : float (None)
            The amount to rotate in latitude (radians).
        lon2 : float (None)
            The amount to rotate the pole in longitude (radians).
        """
        if lon is None:
            lon = np.random.rand()*np.pi*2
        if lat is None:
            # Make sure latitude points spread correctly
            # http://mathworld.wolfram.com/SpherePointPicking.html
            lat = np.arccos(2.*np.random.rand() - 1.)
        if lon2 is None:
            lon2 = np.random.rand()*np.pi*2
        # rotate longitude
        ra = (self.fields_init['RA'] + lon) % (2.*np.pi)
        dec = self.fields_init['dec'] + 0

        # Now to rotate ra and dec about the x-axis
        x, y, z = thetaphi2xyz(ra, dec+np.pi/2.)
        xp, yp, zp = rotx(lat, x, y, z)
        theta, phi = xyz2thetaphi(xp, yp, zp)
        dec = phi - np.pi/2
        ra = theta + np.pi

        # One more RA rotation
        ra = (ra + lon2) % (2.*np.pi)

        self.fields['RA'] = ra
        self.fields['dec'] = dec
        # Rebuild the kdtree with the new positions
        # XXX-may be doing some ra,dec to conversions xyz more than needed.
        self._hp2fieldsetup(ra, dec)
        for bf in self.basis_functions:
            if 'hp2fields' in bf.condition_features:
                bf.condition_features['hp2fields'].update_conditions({'hp2fields': self.hp2fields})

        # self.update_conditions({'hp2fields': self.hp2fields})

    def _update_conditions(self, conditions, **kwargs):
        for bf in self.basis_functions:
            bf.update_conditions(conditions, **kwargs)

        for feature in self.extra_features:
            if hasattr(self.extra_features[feature], 'update_conditions'):
                self.extra_features[feature].update_conditions(conditions, **kwargs)

    def update_conditions(self, conditions, **kwargs):
        self._update_conditions(conditions=conditions, **kwargs)
        # If we are dithering and need to spin the fields
        if self.dither:
            if self.extra_features['night'].feature != self.night:
                self._spin_fields()
                self.night = self.extra_features['night'].feature + 0
        self.reward_checked = False

    def add_observation(self, observation, **kwargs):
        # ugh, I think here I have to assume observation is an array and not a dict.

        if self.ignore_obs not in str(observation['note']):
            for bf in self.basis_functions:
                bf.add_observation(observation, **kwargs)
            for feature in self.extra_features:
                if hasattr(self.extra_features[feature], 'add_observation'):
                    self.extra_features[feature].add_observation(observation, **kwargs)
            self.reward_checked = False

    def __call__(self):
        """
        Just point at the highest reward healpix
        """
        if not self.reward_checked:
            self.reward = self.calc_reward_function()
        # Let's find the best N from the fields
        order = np.argsort(self.reward.data)[::-1]

        iter = 0
        while True:
            best_hp = order[iter*self.block_size:(iter+1)*self.block_size]
            best_fields = np.unique(self.hp2fields[best_hp])
            observations = []
            for field in best_fields:
                if self.tag_fields:
                    tag = np.unique(self.tag_map[np.where(self.hp2fields == field)])[0]
                else:
                    tag = 1
                if tag == 0:
                    continue
                obs = empty_observation()
                obs['RA'] = self.fields['RA'][field]
                obs['dec'] = self.fields['dec'][field]
                obs['rotSkyPos'] = 0.
                obs['filter'] = self.filtername
                obs['nexp'] = 2.  # FIXME: hardcoded
                obs['exptime'] = 30.  # FIXME: hardcoded
                obs['field_id'] = -1
                obs['note'] = self.survey_name
                if self.tag_fields:
                    obs['survey_id'] = np.unique(self.tag_map[np.where(self.hp2fields == field)])[0]
                else:
                    obs['survey_id'] = 1

                observations.append(obs)
                break
            iter += 1
            if len(observations) > 0 or (iter+2)*self.block_size > len(order):
                break

        return observations
