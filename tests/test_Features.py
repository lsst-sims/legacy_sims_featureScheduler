import numpy as np
import unittest
import lsst.sims.featureScheduler.features as features
from lsst.sims.featureScheduler.utils import empty_observation
import lsst.utils.tests


class TestFeatures(unittest.TestCase):

    def testPair_in_night(self):
        pin = features.Pair_in_night(gap_min=25., gap_max=45.)
        self.assertEqual(np.max(pin.feature), 0.)

        indx = np.array([1000])

        delta = 30./60./24.

        # Add 1st observation, feature should still be zero
        obs = empty_observation()
        obs['filter'] = 'r'
        obs['mjd'] = 59000.
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 0.)

        # Add 2nd observation
        obs['mjd'] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 1.)

        obs['mjd'] += delta
        pin.add_observation(obs, indx=indx)
        self.assertEqual(np.max(pin.feature), 2.)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
