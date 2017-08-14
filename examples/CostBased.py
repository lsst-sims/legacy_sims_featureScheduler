import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory

if __name__ == "__main__":

    survey_length = 1  # days
    # Define what we want the final visit ratio map to look like
    survey_filters = ['u','g']
    surveys = []

    for f in survey_filters:
        bfs = []
        bfs.append(fs.Slewtime_basis_function_cost(filtername=f))
        bfs.append(fs.Visit_repeat_basis_function_cost(filtername=f,survey_filters=survey_filters))
        bfs.append(fs.Target_map_basis_function_cost(filtername=f, survey_filters=survey_filters))
        bfs.append(fs.Normalized_alt_basis_function_cost(filtername=f))
        bfs.append(fs.Hour_angle_basis_function_cost())
        #bfs.append(fs.Depth_percentile_basis_function_cost())
        weights = np.array([3,1,1,2,1])
        surveys.append(fs.Simple_greedy_survey_fields_cost(bfs, weights, filtername=f, block_size= 5))

    scheduler = fs.Core_scheduler_cost(surveys)
    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='pairs_survey.db', delete_past=True)
