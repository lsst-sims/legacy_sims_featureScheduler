from lsst.sims.utils import _raDec2Hpid, _approx_RaDec2AltAz
import numpy as np
from lsst.sims.featureScheduler.utils import approx_altaz2pa

__all__ = ["Base_detailer", "Zero_rot_detailer"]


class Base_detailer(object):
    """
    A Detailer is an object that takes a list of proposed observations and adds "details" to them. The
    primary purpose is that the Markov Decision Process does an excelent job selecting RA,Dec,filter
    combinations, but we may want to add additional logic such as what to set the camera rotation angle
    to, or what to use for an exposure time. We could also modify the order of the proposed observations.
    For Deep Drilling Fields, a detailer could be useful for computing dither positions and modifying
    the exact RA,Dec positions.
    """

    def __init__(self, nside=32):
        """
        """
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside

    def add_observation(self, observation, indx=None):
        """
        Parameters
        ----------
        observation : np.array
            An array with information about the input observation
        indx : np.array
            The indices of the healpix map that the observation overlaps with
        """
        for feature in self.survey_features:
            self.survey_features[feature].add_observation(observation, indx=indx)

    def __call__(self, observation_list, conditions):
        """
        Parameters
        ----------
        observation_list : list of observations
            The observations to detail.
        conditions : lsst.sims.featureScheduler.conditions object

        Returns
        -------
        List of observations.
        """

        return observation_list


class Zero_rot_detailer(Base_detailer):
    """
    Detailer to set the camera rotation to be apporximately zero in rotTelPos.
    Because it can never be written too many times:
    rotSkyPos = rotTelPos - ParallacticAngle
    But, wait, what? Is it really the other way?
    """

    def __init__(self, nside=32):
        """
        """
        # Dict to hold all the features we want to track
        self.survey_features = {}
        self.nside = nside

    def __call__(self, observation_list, conditions):

        # XXX--should I convert the list into an array and get rid of this loop?
        for obs in observation_list:
            alt, az = _approx_RaDec2AltAz(obs['RA'], obs['dec'], conditions.site.latitude_rad,
                                          conditions.site.longitude_rad, conditions.mjd)
            obs_pa = approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs['rotSkyPos'] = obs_pa

        return observation_list
