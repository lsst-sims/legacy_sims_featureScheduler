import numpy as np
from lsst.sims.featureScheduler.utils import (empty_observation, set_default_nside)
import healpy as hp
from lsst.sims.featureScheduler.surveys import BaseMarkovDF_survey
from lsst.sims.featureScheduler.utils import (int_binned_stat, max_reject,
                                              gnomonic_project_toxy, tsp_convex)
import copy
from lsst.sims.utils import _angularSeparation, _hpid2RaDec, _approx_RaDec2AltAz


__all__ = ['Greedy_survey', 'Blob_survey']


class Greedy_survey(BaseMarkovDF_survey):
    """
    Select pointings in a greedy way using a Markov Decision Process.
    """
    def __init__(self, basis_functions, basis_weights, filtername='r',
                 block_size=1, smoothing_kernel=None, nside=None,
                 dither=True, seed=42, ignore_obs='ack', survey_name='',
                 nexp=2, exptime=30.):

        extra_features = {}

        super(Greedy_survey, self).__init__(basis_functions=basis_functions,
                                            basis_weights=basis_weights,
                                            extra_features=extra_features,
                                            smoothing_kernel=smoothing_kernel,
                                            ignore_obs=ignore_obs,
                                            nside=nside,
                                            survey_name=survey_name, dither=dither)
        self.filtername = filtername
        self.block_size = block_size
        self.nexp = nexp
        self.exptime = exptime

    def __call__(self, conditions):
        """
        Just point at the highest reward healpix
        """
        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields()
            self.night = conditions.night.copy()

        # Let's find the best N from the fields
        order = np.argsort(self.reward)[::-1]
        # Crop off any NaNs
        order = order[~np.isnan(self.reward[order])]

        iter = 0
        while True:
            best_hp = order[iter*self.block_size:(iter+1)*self.block_size]
            best_fields = np.unique(self.hp2fields[best_hp])
            observations = []
            for field in best_fields:
                obs = empty_observation()
                obs['RA'] = self.fields['RA'][field]
                obs['dec'] = self.fields['dec'][field]
                obs['rotSkyPos'] = 0.
                obs['filter'] = self.filtername
                obs['nexp'] = self.nexp
                obs['exptime'] = self.exptime
                obs['field_id'] = -1
                obs['note'] = self.survey_name

                observations.append(obs)
                break
            iter += 1
            if len(observations) > 0 or (iter+2)*self.block_size > len(order):
                break

        return observations


class Blob_survey(Greedy_survey):
    """Select observations in large, mostly contiuguous, blobs.
    """
    def __init__(self, basis_functions, basis_weights,
                 filtername1='r', filtername2='g',
                 slew_approx=7.5, filter_change_approx=140.,
                 read_approx=2., exptime=30., nexp=2,
                 ideal_pair_time=22., min_pair_time=15.,
                 search_radius=30., alt_max=85., az_range=90.,
                 flush_time=30.,
                 smoothing_kernel=None, nside=None,
                 dither=True, seed=42, ignore_obs='ack',
                 survey_note='blob',
                 sitename='LSST'):
        """
        Parameters
        ----------
        filtername1 : str ('r')
            The filter to observe in.
        filtername2 : str ('g')
            The filter to pair with the first observation. If set to None, no pair
            will be observed.
        slew_approx : float (7.5)
            The approximate slewtime between neerby fields (seconds). Used to calculate
            how many observations can be taken in the desired time block.
        filter_change_approx : float (140.)
             The approximate time it takes to change filters (seconds).
        ideal_pair_time : float (22.)
            The ideal time gap wanted between observations to the same pointing (minutes)
        min_pair_time : float (15.)
            The minimum acceptable pair time (minutes)
        search_radius : float (30.)
            The radius around the reward peak to look for additional potential pointings (degrees)
        alt_max : float (85.)
            The maximum altitude to include (degrees).
        az_range : float (90.)
            The range of azimuths to consider around the peak reward value (degrees).
        flush_time : float (30.)
            The time past the final expected exposure to flush the queue. Keeps observations
            from lingering past when they should be executed. (minutes)
        sitename : str ('LSST')
            The name of the site to lookup latitude and longitude.
        """

        if nside is None:
            nside = set_default_nside()

        super(Blob_survey, self).__init__(basis_functions=basis_functions,
                                          basis_weights=basis_weights,
                                          filtername=None,
                                          block_size=0, smoothing_kernel=smoothing_kernel,
                                          dither=dither, seed=seed, ignore_obs=ignore_obs,
                                          nside=nside)
        self.flush_time = flush_time/60./24.  # convert to days
        self.nexp = nexp
        self.exptime = exptime
        self.slew_approx = slew_approx
        self.read_approx = read_approx
        self.hpids = np.arange(hp.nside2npix(self.nside))
        # If we are taking pairs in same filter, no need to add filter change time.
        if filtername1 == filtername2:
            filter_change_approx = 0
        # Compute the minimum time needed to observe a blob (or observe, then repeat.)
        if filtername2 is not None:
            self.time_needed = (min_pair_time*60.*2. + exptime + read_approx + filter_change_approx)/24./3600.  # Days
        else:
            self.time_needed = (min_pair_time*60. + exptime + read_approx)/24./3600.  # Days
        self.filter_set = set(filtername1)
        if filtername2 is None:
            self.filter2_set = self.filter_set
        else:
            self.filter2_set = set(filtername2)

        self.ra, self.dec = _hpid2RaDec(self.nside, self.hpids)

        self.survey_note = survey_note
        self.counter = 1  # start at 1, because 0 is default in empty observation
        self.filtername1 = filtername1
        self.filtername2 = filtername2
        self.search_radius = np.radians(search_radius)
        self.az_range = np.radians(az_range)
        self.alt_max = np.radians(alt_max)
        self.min_pair_time = min_pair_time
        self.ideal_pair_time = ideal_pair_time

        # If we are only using one filter, this could be useful
        if (self.filtername2 is None) | (self.filtername1 == self.filtername2):
            self.filtername = self.filtername1

    def _set_block_size(self, conditions):
        """
        Update the block size if it's getting near the end of the night.
        """

        available_time = conditions.next_twilight_start - conditions.mjd
        available_time *= 24.*60.  # to minutes

        n_ideal_blocks = available_time / self.ideal_pair_time
        if n_ideal_blocks >= 3:
            self.nvisit_block = int(np.floor(self.ideal_pair_time*60. / (self.slew_approx + self.exptime +
                                                                         self.read_approx*(self.nexp - 1))))
        else:
            # Now we can stretch or contract the block size to allocate the remainder time until twilight starts
            # We can take the remaining time and try to do 1,2, or 3 blocks.
            possible_times = available_time / np.arange(1, 4)
            diff = np.abs(self.ideal_pair_time-possible_times)
            best_block_time = np.max(possible_times[np.where(diff == np.min(diff))])
            self.nvisit_block = int(np.floor(best_block_time*60. / (self.slew_approx + self.exptime +
                                                                    self.read_approx*(self.nexp - 1))))

    def calc_reward_function(self, conditions):
        """
        """
        # Set the number of observations we are going to try and take
        self._set_block_size(conditions)
        #  Computing reward like usual with basis functions and weights
        if self._check_feasibility(conditions):
            self.reward = 0
            indx = np.arange(hp.nside2npix(self.nside))
            for bf, weight in zip(self.basis_functions, self.basis_weights):
                basis_value = bf(conditions, indx=indx)
                self.reward += basis_value*weight
                # might be faster to pull this out into the feasabiliity check?

            if self.smoothing_kernel is not None:
                self.smooth_reward()

            # Apply max altitude cut
            too_high = np.where(conditions.alt > self.alt_max)
            self.reward[too_high] = np.nan

            # Select healpixels within some radius of the max
            # This is probably faster with a kd-tree.
            peak_reward = np.min(np.where(self.reward == np.nanmax(self.reward))[0])

            # Apply radius selection
            dists = _angularSeparation(self.ra[peak_reward], self.dec[peak_reward], self.ra, self.dec)
            out_hp = np.where(dists > self.search_radius)
            self.reward[out_hp] = np.nan

            # Apply az cut
            az_centered = conditions.az - conditions.az[peak_reward]
            az_centered[np.where(az_centered < 0)] += 2.*np.pi

            az_out = np.where((az_centered > self.az_range/2.) & (az_centered < 2.*np.pi-self.az_range/2.))
            self.reward[az_out] = np.nan
        else:
            self.reward = -np.inf
        self.reward_checked = True
        return self.reward

    def __call__(self, conditions):
        """
        Find a good block of observations.
        """

        self.reward = self.calc_reward_function(conditions)

        # Check if we need to spin the tesselation
        if self.dither & (conditions.night != self.night):
            self._spin_fields()
            self.night = conditions.night.copy()

        # Now that we have the reward map,
        potential_hp = np.where(~np.isnan(self.reward) == True)
        # Find the max reward for each potential pointing
        ufields, reward_by_field = int_binned_stat(self.hp2fields[potential_hp],
                                                   self.reward[potential_hp],
                                                   statistic=max_reject)
        order = np.argsort(reward_by_field)
        order = order[~np.isnan(ufields[order])]
        ufields = ufields[order][::-1][0:self.nvisit_block]
        self.best_fields = ufields

        # Let's find the alt, az coords of the points (right now, hopefully doesn't change much in time block)
        pointing_alt, pointing_az = _approx_RaDec2AltAz(self.fields['RA'][self.best_fields],
                                                        self.fields['dec'][self.best_fields],
                                                        conditions.site.latitude_rad,
                                                        conditions.site.longitude_rad,
                                                        conditions.mjd,
                                                        lmst=conditions.lmst)

        # Let's find a good spot to project the points to a plane
        mid_alt = (np.max(pointing_alt) - np.min(pointing_alt))/2.

        # Code snippet from MAF for computing mean of angle accounting for wrap around
        # XXX-TODO: Maybe move this to sims_utils as a generally useful snippet.
        x = np.cos(pointing_az)
        y = np.sin(pointing_az)
        meanx = np.mean(x)
        meany = np.mean(y)
        angle = np.arctan2(meany, meanx)
        radius = np.sqrt(meanx**2 + meany**2)
        mid_az = angle % (2.*np.pi)
        if radius < 0.1:
            mid_az = np.pi

        # Project the alt,az coordinates to a plane. Could consider scaling things to represent
        # time between points rather than angular distance.
        pointing_x, pointing_y = gnomonic_project_toxy(pointing_az, pointing_alt, mid_az, mid_alt)
        # Now I have a bunch of x,y pointings. Drop into TSP solver to get an effiencent route
        towns = np.vstack((pointing_x, pointing_y)).T
        # Leaving optimize=False for speed. The optimization step doesn't usually improve much.
        better_order = tsp_convex(towns, optimize=False)
        # XXX-TODO: Could try to roll better_order to start at the nearest/fastest slew from current position.
        observations = []
        counter2 = 0
        approx_end_time = np.size(better_order)*(self.slew_approx + self.exptime +
                                                 self.read_approx*(self.nexp - 1))
        flush_time = conditions.mjd + approx_end_time/3600./24. + self.flush_time
        for indx in better_order:
            field = self.best_fields[indx]
            obs = empty_observation()
            obs['RA'] = self.fields['RA'][field]
            obs['dec'] = self.fields['dec'][field]
            obs['rotSkyPos'] = 0.
            obs['filter'] = self.filtername1
            obs['nexp'] = self.nexp
            obs['exptime'] = self.exptime
            obs['field_id'] = -1
            obs['note'] = '%s' % (self.survey_note)
            obs['block_id'] = self.counter
            obs['flush_by_mjd'] = flush_time
            # Add the mjd for debugging
            obs['mjd'] = conditions.mjd
            observations.append(obs)
            counter2 += 1

        # If we only want one filter block
        if self.filtername2 is None:
            result = observations
        else:
            # Double the list to get a pair.
            observations_paired = []
            for observation in observations:
                obs = copy.copy(observation)
                obs['filter'] = self.filtername2
                observations_paired.append(obs)

            # Check loaded filter here to decide which goes first
            if conditions.current_filter == self.filtername2:
                result = observations_paired + observations
            else:
                result = observations + observations_paired

            # Let's tag which one is supposed to be first/second in the pair:
            for i in range(0, int(np.size(result)/2), 1):
                result[i]['note'] = '%s, a' % (self.survey_note)
            for i in range(int(np.size(result)/2), np.size(result), 1):
                result[i]['note'] = '%s, b' % (self.survey_note)

        return result
