import numpy as np
import unittest
import lsst.sims.featureScheduler.basis_functions as bf
from lsst.sims.featureScheduler.utils import standard_goals, sim_runner, calc_norm_factor
from lsst.sims.featureScheduler.surveys import (generate_dd_surveys, Greedy_survey,
                                                Blob_survey, Pairs_survey_scripted)
from lsst.sims.featureScheduler import Core_scheduler
import lsst.utils.tests
import healpy as hp
from lsst.sims.speedObservatory import Speed_observatory


def gen_greedy_surveys(nside):
    """
    Make a quick set of greedy surveys
    """
    target_map = standard_goals(nside=nside)
    filters = ['g', 'r', 'i', 'z', 'y']
    surveys = []
    cloud_map = target_map['r'][0]*0 + 0.7

    for filtername in filters:
        bfs = []
        bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=np.nan, nside=nside))
        bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Strict_filter_basis_function(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=60., max_alt=76.))
        bfs.append(bf.Moon_avoidance_basis_function(nside=nside, moon_distance=40.))
        bfs.append(bf.Bulk_cloud_basis_function(max_cloud_map=cloud_map, nside=nside))

        bfs.append(bf.Filter_loaded_basis_function(filternames=filtername))

        weights = np.array([3.0, 0.3, 3., 3., 0., 0., 0., 0.])
        surveys.append(Greedy_survey(bfs, weights, block_size=1, filtername=filtername,
                                     dither=True, nside=nside))
    return surveys


def gen_blob_surveys(nside):
    """
    make a quick set of blob surveys
    """
    target_map = standard_goals(nside=nside)
    norm_factor = calc_norm_factor(target_map)
    cloud_map = target_map['r'][0]*0 + 0.7

    filter1s = ['u', 'g']  # , 'r', 'i', 'z', 'y']
    filter2s = [None, 'g']  # , 'r', 'i', None, None]
    filter1s = ['g']  # , 'r', 'i', 'z', 'y']
    filter2s = ['g']  # , 'r', 'i', None, None]

    pair_surveys = []
    for filtername, filtername2 in zip(filter1s, filter2s):
        bfs = []
        bfs.append(bf.M5_diff_basis_function(filtername=filtername, nside=nside))
        if filtername2 is not None:
            bfs.append(bf.M5_diff_basis_function(filtername=filtername2, nside=nside))
        bfs.append(bf.Target_map_basis_function(filtername=filtername,
                                                target_map=target_map[filtername],
                                                out_of_bounds_val=np.nan, nside=nside,
                                                norm_factor=norm_factor))
        if filtername2 is not None:
            bfs.append(bf.Target_map_basis_function(filtername=filtername2,
                                                    target_map=target_map[filtername2],
                                                    out_of_bounds_val=np.nan, nside=nside,
                                                    norm_factor=norm_factor))
        bfs.append(bf.Slewtime_basis_function(filtername=filtername, nside=nside))
        bfs.append(bf.Strict_filter_basis_function(filtername=filtername))
        # Masks, give these 0 weight
        bfs.append(bf.Zenith_shadow_mask_basis_function(nside=nside, shadow_minutes=60., max_alt=76.))
        bfs.append(bf.Moon_avoidance_basis_function(nside=nside, moon_distance=40.))
        bfs.append(bf.Bulk_cloud_basis_function(max_cloud_map=cloud_map, nside=nside))
        # feasibility basis fucntions. Also give zero weight.
        filternames = [fn for fn in [filtername, filtername2] if fn is not None]
        bfs.append(bf.Filter_loaded_basis_function(filternames=filternames))
        bfs.append(bf.Time_to_twilight_basis_function(time_needed=22.))
        bfs.append(bf.Not_twilight_basis_function())

        weights = np.array([3.0, 3.0, .3, .3, 3., 3., 0., 0., 0., 0., 0., 0.])
        if filtername2 is None:
            # Need to scale weights up so filter balancing still works properly.
            weights = np.array([6.0, 0.6, 3., 3., 0., 0., 0., 0., 0., 0.])
        if filtername2 is None:
            survey_name = 'blob, %s' % filtername
        else:
            survey_name = 'blob, %s%s' % (filtername, filtername2)
        pair_surveys.append(Blob_survey(bfs, weights, filtername1=filtername, filtername2=filtername2,
                                        survey_note=survey_name, ignore_obs='DD'))
    return pair_surveys


class TestFeatures(unittest.TestCase):

    def testGreedy(self):
        """
        Set up a greedy survey and run for a few days. A crude way to touch lots of code.
        """
        nside = 32
        survey_length = 2.1  # days

        surveys = gen_greedy_surveys(nside)
        surveys.append(Pairs_survey_scripted(None, ignore_obs='DD'))

        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.extend(dd_surveys)

        scheduler = Core_scheduler(surveys, nside=nside)
        observatory = Speed_observatory(nside=nside)
        observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                          survey_length=survey_length,
                                                          filename=None)

        # Check that a second part of a pair was taken
        assert('pair(scripted)' in observations['note'])
        # Check that the a DD was observed
        assert('DD:ECDFS' in observations['note'])
        # And the u-band
        assert('DD:u,ECDFS' in observations['note'])
        # Make sure a few different filters were observed
        assert(len(np.unique(observations['filter'])) > 3)
        # Make sure lots of observations executed
        assert(observations.size > 1000)
        # Make sure nothing tried to look through the earth
        assert(np.min(observations['alt']) > 0)

    def testBlobs(self):
        """
        Set up a blob selection survey
        """
        nside = 32
        survey_length = 2.1  # days

        surveys = []
        # Set up the DD
        dd_surveys = generate_dd_surveys(nside=nside)
        surveys.append(dd_surveys)

        surveys.append(gen_blob_surveys(nside))
        surveys.append(gen_greedy_surveys(nside))

        scheduler = Core_scheduler(surveys, nside=nside)
        observatory = Speed_observatory(nside=nside)
        observatory, scheduler, observations = sim_runner(observatory, scheduler,
                                                          survey_length=survey_length,
                                                          filename=None)

        # Make sure some blobs executed
        assert('blob, gg, a' in observations['note'])
        assert('blob, gg, b' in observations['note'])
        # assert('blob, u' in observations['note'])

        # Make sure some greedy executed
        assert('' in observations['note'])
        # Check that the a DD was observed
        assert('DD:ECDFS' in observations['note'])
        # And the u-band
        assert('DD:u,ECDFS' in observations['note'])
        # Make sure a few different filters were observed
        assert(len(np.unique(observations['filter'])) > 3)
        # Make sure lots of observations executed
        assert(observations.size > 1000)
        # Make sure nothing tried to look through the earth
        assert(np.min(observations['alt']) > 0)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
