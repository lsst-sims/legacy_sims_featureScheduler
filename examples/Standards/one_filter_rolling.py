import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import healpy as hp
import lsst.sims.utils as utils


if __name__ == "__main__":

    nside = 64
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    ra, dec = utils.hpid2RaDec(nside, np.arange(mask.size))
    mregion = np.where((dec > -35) & (dec < -10))
    mask[mregion] = True

    for survey_length in [365.25, 365.25*2]:
        year = np.round(survey_length/365.25)
        # Define what we want the final visit ratio map to look like
        target_map = fs.standard_goals()['r']
        filtername = 'r'

        bfs = []
        bfs.append(fs.M5_diff_basis_function(filtername=filtername))
        bfs.append(fs.Target_map_basis_function(target_map=target_map, filtername=filtername,
                                                out_of_bounds_val= hp.UNSEEN))
        bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50.))
        bfs.append(fs.Slewtime_basis_function(filtername=filtername))
        bfs.append(fs.Rolling_mask_basis_function(mask=mask, mjd_start=59580.035))

        weights = np.array([1., 0.2, 1., 2., 1.])
        survey = fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=True)
        scheduler = fs.Core_scheduler([survey])

        observatory = Speed_observatory()
        observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                             survey_length=survey_length,
                                                             filename='rolling_%i.db' % year,
                                                             delete_past=True)

