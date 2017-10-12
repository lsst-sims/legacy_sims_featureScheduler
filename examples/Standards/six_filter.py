import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import healpy as hp

# Run a single-filter r-band survey.
# 5-sigma depth percentile
# standard target map (WFD, NES, SCP, GP)
# Slewtime
# mask lots of off-meridian space
# No pairs
# Greedy selection of opsim fields


if __name__ == "__main__":

    survey_length = 365.25  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    surveys = []

    for filtername in filters:

        bfs = []
        bfs.append(fs.Depth_percentile_basis_function(filtername=filtername))
        bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=hp.UNSEEN))

        bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50.))
        bfs.append(fs.Slewtime_basis_function(filtername=filtername))
        bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

        weights = np.array([2., 0.2, 1., 1., 3.])
        surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername))

    scheduler = fs.Core_scheduler(surveys)

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='six_filter.db',
                                                         delete_past=True)
# real    560m57.716s
