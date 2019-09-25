import numpy as np
from lsst.sims.featureScheduler.detailers import Base_detailer
from lsst.sims.utils import _approx_RaDec2AltAz
from lsst.sims.featureScheduler.utils import approx_altaz2pa


__all__ = ["Dither_detailer", "Camera_rot_detailer"]


def gnomonic_project_toxy(ra, dec, raCen, decCen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
    Input radians. Returns x/y."""
    # also used in Global Telescope Network website
    if (len(ra) != len(dec)):
        raise Exception("Expect RA and Dec arrays input to gnomonic projection to be same length.")
    cosc = np.sin(decCen) * np.sin(dec) + np.cos(decCen) * np.cos(dec) * np.cos(ra-raCen)
    x = np.cos(dec) * np.sin(ra-raCen) / cosc
    y = (np.cos(decCen)*np.sin(dec) - np.sin(decCen)*np.cos(dec)*np.cos(ra-raCen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, raCen, decCen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(decCen) - y * np.sin(decCen)
    ra = raCen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(decCen) + y * np.cos(decCen), np.sqrt(x*x + denom*denom))
    return ra, dec


class Dither_detailer(Base_detailer):
    """
    make a uniform dither pattern. Offset by a maximum radius in a random direction.

    Parameters
    ----------
    max_dither : float (0.7)
        The maximum dither size to use (degrees).
    per_night : bool (True)
        If true, us the same dither offset for an entire night


    """
    def __init__(self, max_dither=0.7, seed=42, per_night=True):
        self.survey_features = {}

        self.current_night = -1
        self.max_dither = np.radians(max_dither)
        self.per_night = per_night
        np.random.seed(seed=seed)
        self.offset = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = (np.random.random((1, 2))-0.5) * 2.*self.max_dither
                angle = np.random.random(1)*2*np.pi
                radius = self.max_dither * np.sqrt(np.random.random(1))
                self.offset = np.array([radius*np.cos(angle), radius*np.sin(angle)])
            offsets = np.tile(self.offset, (n_offsets, 1))
        else:
            angle = np.random.random(n_offsets)*2*np.pi
            radius = self.max_dither * np.sqrt(np.random.random(n_offsets))
            offsets = np.array([radius*np.cos(angle), radius*np.sin(angle)])

        return offsets

    def __call__(self, observation_list, conditions):

        # Generate offsets in RA and Dec
        offsets = self._generate_offsets(len(observation_list), conditions.night)
        
        obs_array = np.concatenate(observation_list)
        newRA, newDec = gnomonic_project_tosky(offsets[0, :], offsets[1, :], obs_array['RA'], obs_array['dec'])
        for i, obs in enumerate(observation_list):
            observation_list[i]['RA'] = newRA[i]
            observation_list[i]['dec'] = newDec[i]
        return observation_list


class Camera_rot_detailer(Base_detailer):
    """
    Randomly set the camera rotation, either for each exposure, or per night.

    Parameters
    ----------
    max_rot : float (90.)
        The maximum amount to offset the camera (degrees)
    min_rot : float (90)
        The minimum to offset the camera (degrees)
    per_night : bool (True)
        If True, only set a new offset per night. If False, randomly rotates every observation.
    """
    def __init__(self, max_rot=90., min_rot=-90., per_night=True, seed=42):
        self.survey_features = {}

        self.current_night = -1
        self.max_rot = np.radians(max_rot)
        self.min_rot = np.radians(min_rot)
        self.range = self.max_rot - self.min_rot
        self.per_night = per_night
        np.random.seed(seed=seed)
        self.offset = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = np.random.random(1) * self.range - self.min_rot
            offsets = np.ones(n_offsets) * self.offset
        else:
            offsets = np.random.random(n_offsets) * self.range - self.min_rot

        return offsets

    def __call__(self, observation_list, conditions):

        # Generate offsets in RA and Dec
        offsets = self._generate_offsets(len(observation_list), conditions.night)

        for i, obs in enumerate(observation_list):
            alt, az = _approx_RaDec2AltAz(obs['RA'], obs['dec'], conditions.site.latitude_rad,
                                          conditions.site.longitude_rad, conditions.mjd)
            obs_pa = approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs['rotSkyPos'] = (obs_pa + offsets[i] + 2.*np.pi) % (2.*np.pi)

        return observation_list
