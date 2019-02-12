import numpy as np
import matplotlib.pylab as plt
import healpy as hp
from lsst.sims.featureScheduler.modelObservatory import Model_observatory
from lsst.sims.featureScheduler.schedulers import Core_scheduler
from lsst.sims.featureScheduler.utils import standard_goals, calc_norm_factor
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.surveys import (generate_dd_surveys, Greedy_survey,
                                                Blob_survey, Pairs_survey_scripted)
from lsst.sims.featureScheduler import sim_runner


def gen_greedy_surveys(nside, m5_weight=3., count_uniformity_weight=0.3,
                       slewtime_weight=3., filter_change_weight=4.,
                       filters = ['u', 'g', 'r', 'i', 'z', 'y'], add_DD=False,
                       pairs=False):
    """
    Make a quick set of greedy surveys
    """
    sg = standard_goals(nside=nside)
    target_map = {}
    for key in filters:
        target_map[key] = sg[key]
    norm_factor = calc_norm_factor(target_map)
    surveys = []

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
        bfs.append(bf.Clouded_out_basis_function())

        bfs.append(bf.Filter_loaded_basis_function(filternames=filtername))

        weights = np.array([m5_weight, count_uniformity_weight, slewtime_weight, filter_change_weight,
                           0., 0., 0., 0.])
        surveys.append(Greedy_survey(bfs, weights, block_size=1, filtername=filtername,
                                     dither=True, nside=nside, ignore_obs='DD'))

    if pairs:
        surveys.append(Pairs_survey_scripted(None, ignore_obs='DD'))
    if add_DD:
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.extend(dd_surveys)

    return surveys
