import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.observatory import Speed_observatory


if __name__ == "__main__":

    survey_length = 2.  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['r']
    # Make a list of the basis functions we want to use
    bfs = []
    # Reward looking at parts of the sky with good 5-sigma depth
    bfs.append(fs.Depth_percentile_basis_function(filtername='r'))
    # Reward parts of the survey that have fallen behind
    bfs.append(fs.Target_map_basis_function(target_map=target_map))
    # Reward smaller slewtimes
    bfs.append(fs.Slewtime_basis_function(filtername='r'))

    weights = np.array([10., 1., 10.])
    # Use just the opsim fields for simplicity
    survey = fs.Simple_greedy_survey_fields(bfs, weights)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='one_filter.db')
