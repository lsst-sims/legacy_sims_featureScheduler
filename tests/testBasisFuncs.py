import numpy as np
import unittest
import lsst.sims.featureScheduler as fs
import lsst.utils.tests


class TestBasis(unittest.TestCase):

    def testVisit_repeat_basis_function(self):
        bf = fs.Visit_repeat_basis_function()

        indx = np.array([1000])

        # 20 minute step
        delta = 20./60./24.

        # Add 1st observation, should still be zero
        obs = fs.empty_observation()
        obs['filter'] = 'r'
        obs['mjd'] = 59000.
        conditions = {'mjd': obs['mjd']}
        bf.add_observation(obs, indx=indx)
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 0.)

        conditions['mjd'] += delta
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 1.)

        # Now complete the pair and it should go back to zero
        obs['mjd'] += delta
        bf.add_observation(obs, indx=indx)

        conditions['mjd'] += delta
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 0.)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
