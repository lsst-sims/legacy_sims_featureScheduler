import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.observatory import Speed_observatory
import healpy as hp

if __name__ == "__main__":

    survey_length = 365  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['r']

    bfs = []
    bfs.append(fs.Depth_percentile_basis_function())
    bfs.append(fs.Target_map_basis_function(target_map=target_map))
    bfs.append(fs.Quadrant_basis_function())
    bfs.append(fs.Slewtime_basis_function())

    weights = np.array([.5, 1., 1., 1.])
    survey = fs.Simple_greedy_survey_fields(bfs, weights, block_size=1)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='marching_block_1.db',
                                                         delete_past=True)

# block_size=10, surveylength of 365 had runtime of 163 min. and got 0.26e6 observations. So, 10 years would be 27 hours.
# Going to block_size=1, runtime of 211 min, and 0.33e6 observations. So, 35 hours. Not too shabby! 