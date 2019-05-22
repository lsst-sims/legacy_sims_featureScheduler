from lsst.sims.featureScheduler.detailers import Base_detailer
from lsst.sims.utils import _raDec2Hpid, m5_flat_sed
import lsst.sims.featureScheduler.features as features
from lsst.sims.featureScheduler.utils import hp_in_lsst_fov
import numpy as np
import healpy as hp


class Short_expt_detailer(Base_detailer):
    """Check if the area has been observed with a short exposure time this year. If not, add some short exposures.

    Parameters
    ----------
    exp_time : float (1.)
        The short exposure time to use.

    """
    def __init__(self, exp_time=1., filtername='r', nside=32, footprint=None, nobs=2, survey_name='short'):
        self.exp_time = exp_time
        self.filtername = filtername
        self.nside = nside
        self.footprint = footprint
        self.nobs = nobs
        self.survey_name = survey_name

        self.survey_features = {}
        # XXX--need a feature that tracks short exposures in the filter
        self.survey_features['nobs'] = features.N_observations_currentyear(filtername=filtername, nside=nside,
                                                                           survey_name=self.survey_name)
        # Need to be able to look up hpids for each observation
        self.obs2hpid = hp_in_lsst_fov(nside=nside)

    def __call__(self, observation_list, conditions):
        out_observations = []
        for observation in observation_list:
            out_observations.append(observation)
            if observation['filter'] == self.filtername:
                hpids = self.obs2hpid(observation['RA'], observation['dec'])
                # Crop off anything outside the target footprint
                hpids = hpids[np.where(self.footprint[hpids] > 0)]
                # Crop off things where we already have enough observation
                hpids = hpids[np.where(self.survey_features['nobs'].feature[hpids] < self.nobs)]
                if np.size(hpids) > 0:
                    new_obs = observation.copy()
                    new_obs['exptime'] = self.exp_time
                    new_obs['nexp'] = 1
                    new_obs['note'] = self.survey_name
                    out_observations.append(new_obs)

        return out_observations
