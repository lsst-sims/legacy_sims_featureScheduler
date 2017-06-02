import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.observatory import Speed_observatory


if __name__ == "__main__":

    survey_length = 2.2  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['r']

    bfs = []
    bfs.append(fs.Depth_percentile_basis_function())
    bfs.append(fs.Target_map_basis_function(target_map=target_map))
    bfs.append(fs.Visit_repeat_basis_function())
    bfs.append(fs.Slew_distance_basis_function())
    weights = np.array([2., 1., 2., 1.])
    survey = fs.Simple_greedy_survey_fields(bfs, weights)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='pairs_survey.db')
