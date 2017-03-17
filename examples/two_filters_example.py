import numpy as np
import healpy as hp
import matplotlib.pylab as plt
import lsst.sims.featureScheduler as fs
from speed_observatory import Speed_observatory


if __name__ == "__main__":

    survey_length = 5.2  # days
    # Define what we want the final visit ratio map to look like
    target_maps = fs.standard_goals()

    filters = ['r', 'i']
    weights = {}
    weights['r'] = [1., 1., 1.]
    weights['i'] = [1.5, 1., 0.]
    surveys = []
    for filtername in filters:
        bfs = []
        bfs.append(fs.Depth_percentile_basis_function(filtername=filtername))
        bfs.append(fs.Target_map_basis_function(target_map=target_maps[filtername], filtername=filtername))
        bfs.append(fs.Filter_change_basis_function(filtername=filtername))
        weight = weights[filtername]
        surveys.append(fs.Simple_greedy_survey_fields(bfs, weight, filtername=filtername))

    scheduler = fs.Core_scheduler(surveys)

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler, survey_length=survey_length,
                                                         filename='two_filt.db')
