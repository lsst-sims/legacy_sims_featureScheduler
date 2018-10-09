import numpy as np
import unittest
#import lsst.sims.featureScheduler as fs
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.utils import standard_goals, sim_runner
from lsst.sims.featureScheduler.surveys import generate_dd_surveys, Greedy_survey
from lsst.sims.featureScheduler import Core_scheduler
import lsst.utils.tests
import healpy as hp
from lsst.sims.speedObservatory import Speed_observatory


class TestFeatures(unittest.TestCase):

    def testBaseline(self):
        """
        Set up a baseline survey and run for a few days. A crude way to touch lots of code.
        """
        nside = 32

        survey_length = 2.1  # days

        # Define what we want the final visit ratio map to look like
        target_map = standard_goals(nside=nside)
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        surveys = []

        for filtername in filters:
            bfs = []
            bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
            bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                    target_map=target_map[filtername],
                                                    out_of_bounds_val=hp.UNSEEN, nside=nside))

            #bfs.append(bf.North_south_patch_basis_function(zenith_min_alt=50., nside=nside))
            bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
            bfs.append(bf.Strict_filter_basis_function(filtername=filtername))

            weights = np.array([3.0, 0.3, 3., 3.])
            surveys.append(Greedy_survey(bfs, weights, block_size=1, filtername=filtername,
                                         dither=True, nside=nside))

        # XXX--haven't ported pairs yet
        #surveys.append(bf.Pairs_survey_scripted([], [], ignore_obs='DD'))

        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.extend(dd_surveys)

        scheduler = Core_scheduler(surveys, nside=nside)
        observatory = Speed_observatory(nside=nside)
        observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                          survey_length=survey_length,
                                                          filename=None)

        # Check that a second part of a pair was taken
        #assert('pair(scripted)' in observations['note'])
        
        # Check that the COSMOS DD was observed
        assert('DD:ECDFS' in observations['note'])
        # And the u-band
        assert('DD:u,ECDFS' in observations['note'])
        # Make sure a few different filters were observed
        assert(len(np.unique(observations['filter'])) > 3)
        # Make sure lots of observations executed
        assert(observations.size > 1000)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
