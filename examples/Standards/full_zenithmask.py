import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory

if __name__ == '__main__':
    nside = fs.set_default_nside(nside=32)

    survey_length = 365.25  # *10  # days

    years = np.round(survey_length/365.25)
    target_map = fs.standard_goals()
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    surveys = []

    for filtername in filters:
        bfs = []
        bfs.append(fs.M5_diff_basis_function(filtername=filtername))
        bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername]))

        bfs.append(fs.Zenith_mask_basis_function(max_alt=75., penalty=-20))
        bfs.append(fs.Slewtime_basis_function(filtername=filtername))
        bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

        weights = np.array([2.0, 0.2, 1., 2., 3.])
        surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername,
                                               dither=True, ignore_obs='DD'))

    surveys.append(fs.Pairs_survey_scripted([], [], ignore_obs='DD'))

    # Set up the DD
    surveys.extend(fs.generate_dd_surveys())

    scheduler = fs.Core_scheduler(surveys)

    observatory = Speed_observatory()
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='full_zenithmask_%i.db' % years,
                                                         delete_past=True)
