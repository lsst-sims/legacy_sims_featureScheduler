import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.featureScheduler.observatory import Speed_observatory

if __name__ == "__main__":

    survey_length = 5  # days
    # Define what we want the final visit ratio map to look like
    target_map = fs.standard_goals()['r']
    filters = ['r']
    survey_filters = ['r', 'g']

    bfs = []
    bfs.append(fs.Depth_percentile_basis_function())
    bfs.append(fs.Target_map_basis_function_cost(target_map=target_map, survey_filters = survey_filters))
    bfs.append(fs.Visit_repeat_basis_function_cost(survey_filters = survey_filters))
    bfs.append(fs.Slewtime_basis_function_cost())
    weights = -1*np.array([ 1., 1., 1., 1.])
    survey = fs.Simple_greedy_survey_fields(bfs, weights)
    scheduler = fs.Core_scheduler([survey])

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='pairs_survey.db')


