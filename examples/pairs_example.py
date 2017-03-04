import numpy as np
import lsst.sims.featureScheduler as fs
from speed_observatory import Speed_observatory


if __name__ == "__main__":

    survey_length = 2.  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['r']

    bfs = []
    bfs.append(fs.Depth_percentile_basis_function())
    bfs.append(fs.Target_map_basis_function(target_map=target_map))
    bfs.append(fs.Visit_repeat_basis_function())
    weights = np.array([.5, 1., 1.])
    survey = fs.Simple_greedy_survey(bfs, weights)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='pairs_survey.db')
