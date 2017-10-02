import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import healpy as hp
import matplotlib.pylab as plt


if __name__ == "__main__":

    az_widths = [15., 30.]
    for az_width in az_widths:
        outdir = 'y_march_south_%i' % az_width
        survey_length = 366  # days
        # Define what we want the final visit ratio map to look like
        target_map = fs.standard_goals()['y']
        bfs = []
        # Target number of observations
        bfs.append(fs.Target_map_basis_function(filtername='y', target_map=target_map))
        # Mask everything but the South
        bfs.append(fs.Quadrant_basis_function(quadrants=['S'], azWidth=az_width))
        # throw in the depth percentile for good measure
        bfs.append(fs.Depth_percentile_basis_function())
        weights = np.array([1., 1., 1.])

        survey = fs.Marching_army_survey(bfs, weights, npick=40)
        scheduler = fs.Core_scheduler([survey])

        observatory = Speed_observatory()
        observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                             survey_length=survey_length,
                                                             filename=outdir+'/y_marching_south_%i.db' % az_width)

        title = 'az width %i' % az_width

        hp.mollview(scheduler.surveys[0].basis_functions[0].survey_features['N_obs'].feature,
                    unit='N Visits', title=title)
        plt.savefig(outdir+'/n_viz.pdf')

        plt.figure()
        none = plt.hist(observations['slewtime'], bins=50)
        plt.xlabel('slewtime (seconds)')
        plt.ylabel('Count')
        plt.title('mean = %.2fs' % np.mean(observations['slewtime']))
        plt.savefig(outdir+'/time_hist.pdf')

# real    378m17.093s timing for 2-years
