import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory

if __name__ == "__main__":

    survey_length = 60.  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['y']
    bfs = []
    # Target number of observations
    bfs.append(fs.Target_map_basis_function(target_map=target_map))
    # Mask everything but the South
    bfs.append(fs.Quadrant_basis_function(quadrants=['S']))
    # throw in the depth percentile for good measure
    bfs.append(fs.Depth_percentile_basis_function())
    weights = np.array([1., 1., 1.])

    survey = fs.Marching_army_survey(bfs, weights)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='y_marching_south.db')
    