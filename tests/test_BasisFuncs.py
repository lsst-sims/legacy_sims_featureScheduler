import numpy as np
import unittest
import lsst.sims.featureScheduler as fs
import lsst.utils.tests
import matplotlib.pylab as plt


class TestBasis(unittest.TestCase):

    def testVisit_repeat_basis_function(self):
        bf = fs.Visit_repeat_basis_function()

        indx = np.array([1000])

        # 30 minute step
        delta = 30./60./24.

        # Add 1st observation, should still be zero
        obs = fs.empty_observation()
        obs['filter'] = 'r'
        obs['mjd'] = 59000.
        conditions = {'mjd': obs['mjd']}
        bf.add_observation(obs, indx=indx)
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 0.)

        # Advance time so now we want a pair
        conditions['mjd'] += delta
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 1.)

        # Now complete the pair and it should go back to zero
        bf.add_observation(obs, indx=indx)

        conditions['mjd'] += delta
        bf.update_conditions(conditions)
        self.assertEqual(np.max(bf()), 0.)

    def testHADecAltAzPatchBasisFunction(self):
        """Basic test for the HADecAltAzPatchBasisFunction. Just make sure the default parameter returns a mask
        with observable regions.
        """

        bf = fs.HADecAltAzPatchBasisFunction()

        mjd = 59000.
        altaz_feature = fs.AltAzFeature()

        site = fs.Site(name='LSST')
        lmst, last = fs.calcLmstLast(mjd, site.longitude_rad)
        lmst_feature = fs.Current_lmst()
        conditions = {'altaz': altaz_feature,
                      'lmst': lmst,
                      'mjd': mjd}

        lmst_feature.update_conditions(conditions)
        altaz_feature.update_conditions(conditions=conditions)

        bf.update_conditions(conditions)

        mask = bf()

        # Just check that the default mask has valid values. In the future could add a mask here and make sure they
        # are the same.
        self.assertTrue(np.any(mask == 1.))

    def testMeridianStripeBasisFunction(self):
        """Basic test for the MeridianStripeBasisFunction. Just make sure the default parameter returns a mask
        with observable regions.
        """

        bf = fs.MeridianStripeBasisFunction()

        mjd = 59000.
        altaz_feature = fs.AltAzFeature()

        site = fs.Site(name='LSST')
        lmst, last = fs.calcLmstLast(mjd, site.longitude_rad)
        lmst_feature = fs.Current_lmst()
        conditions = {'altaz': altaz_feature,
                      'lmst': lmst,
                      'mjd': mjd}

        lmst_feature.update_conditions(conditions)
        altaz_feature.update_conditions(conditions=conditions)

        bf.update_conditions(conditions)

        mask = bf()

        # Just check that the default mask has valid values. In the future could add a mask here and make sure they
        # are the same.
        self.assertTrue(np.any(mask == 1.))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
