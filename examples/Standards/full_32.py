import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import matplotlib.pylab as plt
import healpy as hp
from numpy.lib.recfunctions import append_fields

if __name__ == '__main__':
    nside = fs.set_default_nside(nside=32)

    survey_length = 365.25*10  # days
    # Define what we want the final visit ratio map to look like
    years = np.round(survey_length/365.25)
    target_map = fs.standard_goals(nside=nside)
    filters = ['u', 'g', 'r', 'i', 'z', 'y']
    surveys = []

    for filtername in filters:
        bfs = []
        bfs.append(fs.M5_diff_basis_function(filtername=filtername, nside=nside))
        bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=hp.UNSEEN, nside=nside))

        bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50., nside=nside))
        #bfs.append(fs.Zenith_mask_basis_function(maxAlt=78., penalty=-100, nside=nside))
        bfs.append(fs.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

        weights = np.array([3.0, 0.4, 1., 2., 3.])
        surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername,
                                               dither=True, nside=nside))

    surveys.append(fs.Pairs_survey_scripted([], [], ignore_obs='DD'))

    # Set up the DD
    dd_survey = fs.Scripted_survey([], [])
    names = ['RA', 'dec', 'mjd', 'filter']
    types = [float, float, float, '|1U']
    observations = np.loadtxt('minion_dd.csv', skiprows=1, dtype=list(zip(names, types)), delimiter=',')
    exptimes = np.zeros(observations.size)
    exptimes.fill(30.)
    observations = append_fields(observations, 'exptime', exptimes)
    nexp = np.zeros(observations.size)
    nexp.fill(2)
    observations = append_fields(observations, 'nexp', nexp)
    notes = np.zeros(observations.size, dtype='|2U')
    notes.fill('DD')
    observations = append_fields(observations, 'note', notes)
    dd_survey.set_script(observations)
    surveys.append(dd_survey)

    scheduler = fs.Core_scheduler(surveys, nside=nside)
    observatory = Speed_observatory(nside=nside)
    observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                         survey_length=survey_length,
                                                         filename='full_nside32_%i.db' % years,
                                                         delete_past=True)

#  1962m28.940s = 32.7 hr
