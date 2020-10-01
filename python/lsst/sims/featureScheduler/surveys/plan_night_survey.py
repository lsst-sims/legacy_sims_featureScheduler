from lsst.sims.featureScheduler.surveys import Blob_survey
from lsst.sims.featureScheduler import features
import numpy as np
from lsst.sims.featureScheduler.utils import (empty_observation, set_default_nside)
import healpy as hp
import matplotlib.pylab as plt


__all__ = ['Plan_ahead_survey']


class Plan_ahead_survey(Blob_survey):
    """Have a survey object that can plan ahead if it will want to observer a blob later in the night

    Parameters
    ----------
    delta_mjd_tol : float
        The tolerance to alow on when to execute scheduled observations
    minimum_sky_area : float
        The minimum sky area to demand before making a scheduled observation
    """

    def __init__(self, basis_functions, basis_weights, delta_mjd_tol=0.3/24., minimum_sky_area=200.,
                 track_filters='g', in_season=2.5, cadence=9, **kwargs,):
        super(Plan_ahead_survey, self).__init__(basis_functions, basis_weights, **kwargs)
        self.current_night = -100
        self.scheduled_obs = None
        self.delta_mjd_tol = delta_mjd_tol
        self.minimum_sky_area = minimum_sky_area  # sq degrees
        self.extra_features = {}
        self.extra_features['last_observed'] = features.Last_observed(filtername=track_filters)
        self.in_season = in_season/12.*np.pi  # to radians

        self.pix_area = hp.nside2pixarea(self.nside, degrees=True)
        self.cadence = cadence

    def check_night(self, conditions):
        """
        """
        delta_mjd = conditions.mjd - self.extra_features['last_observed'].feature

        pix_to_obs = np.where((delta_mjd > self.cadence) & (np.abs(conditions.az_to_antisun) < self.in_season))[0]

        area = np.size(pix_to_obs)*self.pix_area

        # If there are going to be some observations at a given time
        if area > self.minimum_sky_area:
            # Maybe just calculate the mean (with angles)
            mean_RA = np.arctan2(np.sum(np.sin(conditions.ra[pix_to_obs])), np.sum(np.cos(conditions.dec[pix_to_obs])))

            hour_angle = conditions.lmst - mean_RA*12./np.pi
            if hour_angle < 0:
                hour_angle += 24
            # Now we are running from 0 to 24 hours
            self.scheduled_obs = conditions.mjd + hour_angle/24.
            # Keep in mind that this could be getting called again, so as long as we're still close, it's fine.
        else:
            self.scheduled_obs = None

    def calc_reward_function(self, conditions):
        # Only compute if we will want to observe sometime in the night
        self.reward = -np.inf
        if self.night != conditions.night:
            self.check_night(conditions)
            self.night = conditions.night + 0

        # If there are scheduled observations, and we are in the correct time window
        delta_mjd = conditions.mjd - self.scheduled_obs
        # If we are past when we wanted to execute
        if delta_mjd > self.delta_mjd_tol:
            self.check_night(conditions)

        if self.scheduled_obs is not None:
            if np.abs(delta_mjd) < self.delta_mjd_tol:
                # Check the night again, in case something else went and
                # did the observations we were going to do
                self.check_night(conditions)
                if self.scheduled_obs is not None:
                    if np.abs(delta_mjd) < self.delta_mjd_tol:
                        self.reward = super().calc_reward_function(conditions)

        return self.reward
