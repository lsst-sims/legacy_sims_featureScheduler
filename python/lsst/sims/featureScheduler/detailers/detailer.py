from lsst.sims.utils import _raDec2Hpid, _approx_RaDec2AltAz
import numpy as np
from lsst.sims.featureScheduler.utils import approx_altaz2pa

__all__ = ["Base_detailer", "Zero_rot_detailer", "Comcam_90rot_detailer"]


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


class Comcam_90rot_detailer(Base_detailer):
    """
    Detailer to set the camera rotation so rotSkyPos is 0, 90, 180, or 270 degrees. Whatever
    is closest to rotTelPos of zero.
    """

    def __call__(self, observation_list, conditions):
        favored_rotSkyPos = np.radians([0., 90., 180., 270., 360.]).reshape(5, 1)
        obs_array =np.concatenate(observation_list)
        alt, az = _approx_RaDec2AltAz(obs_array['RA'], obs_array['dec'], conditions.site.latitude_rad,
                                      conditions.site.longitude_rad, conditions.mjd)
        parallactic_angle = approx_altaz2pa(alt, az, conditions.site.latitude_rad)
        # If we set rotSkyPos to parallactic angle, rotTelPos will be zero. So, find the
        # favored rotSkyPos that is closest to PA to keep rotTelPos as close as possible to zero.
        ang_diff = np.abs(parallactic_angle - favored_rotSkyPos)
        min_indxs = np.argmin(ang_diff, axis=0)
        # can swap 360 and zero if needed?
        final_rotSkyPos = favored_rotSkyPos[min_indxs]
        # Set all the observations to the proper rotSkyPos
        for rsp, obs in zip(final_rotSkyPos, observation_list):
            obs['rotSkyPos'] = rsp

        return observation_list






