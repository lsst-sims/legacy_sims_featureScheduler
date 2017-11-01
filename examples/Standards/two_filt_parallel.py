import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import healpy as hp

# Try out the parallel stuff
# start engines with:
# ipcluster start -n 2

if __name__ == "__main__":

    parallel = True

    survey_length = 0.5 #365.25  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()
    filters = ['g', 'r', 'i', 'z']
    surveys = []

    for filtername in filters:

        bfs = []
        bfs.append(fs.M5_diff_basis_function(filtername=filtername))
        bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=hp.UNSEEN))

        bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50.))
        bfs.append(fs.Slewtime_basis_function(filtername=filtername))
        bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

        weights = np.array([3., 0.2, 1., 2., 3.])
        surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername))

    if parallel:
        scheduler = fs.Core_scheduler_parallel(surveys)
    else:
        scheduler = fs.Core_scheduler(surveys)

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='two_filt_par.db',
                                                         delete_past=True)


# Time for single core, 0.5 days:
# 0m54s
# Time for 4 cores, 0.5 days:
# 