import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from lsst.sims.featureScheduler.mockTelem import Mock_observatory
from lsst.sims.featureScheduler.schedulers import Core_scheduler
from lsst.sims.featureScheduler.utils import sim_runner, standard_goals, calc_norm_factor
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.surveys import (generate_dd_surveys, Greedy_survey,
                                                Blob_survey, Pairs_survey_scripted)


def gen_greedy_surveys(nside, add_DD=True):
    """
    Make a quick set of greedy surveys
    """
    target_map = standard_goals(nside=nside)
    norm_factor = calc_norm_factor(target_map)
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    surveys = []
    cloud_map = target_map['r'][0]*0 + 0.7

    for filtername in filters:
        bfs = []
        bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=np.nan, nside=nside,
                                                norm_factor=norm_factor))
        bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Strict_filter_basis_function(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=60., max_alt=76.))
        bfs.append(bf.Moon_avoidance_basis_function(nside=nside, moon_distance=40.))
        bfs.append(bf.Bulk_cloud_basis_function(max_cloud_map=cloud_map, nside=nside))

        bfs.append(bf.Filter_loaded_basis_function(filternames=filtername))

        weights = np.array([3.0, 0.3, 3., 3., 0., 0., 0., 0.])
        surveys.append(Greedy_survey(bfs, weights, block_size=1, filtername=filtername,
                                     dither=True, nside=nside, ignore_obs='DD'))

    surveys.append(Pairs_survey_scripted(None, ignore_obs='DD'))
    if add_DD:
        dd_surveys = generate_dd_surveys(nside=nside)

    surveys.extend(dd_surveys)

    return surveys

if __name__ == "__main__":
    nside = 32
    survey_length = 366  # Days
    years = int(survey_length/365.25)

    surveys = gen_greedy_surveys(nside, add_DD=True)

    n_visit_limit = None
    scheduler = Core_scheduler(surveys, nside=nside)
    observatory = Mock_observatory(nside=nside)
    observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                      survey_length=survey_length,
                                                      filename='greedy_%i.db' % years,
                                                      n_visit_limit=n_visit_limit)
