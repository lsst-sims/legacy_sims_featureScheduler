import numpy as np
import unittest
from lsst.sims.featureScheduler.utils import season_calc, create_season_offset
import lsst.utils.tests
import healpy as hp


class TestFeatures(unittest.TestCase):

    def testSeason(self):
        """
        Test that the season utils work as intended
        """
        night = 365.25 * 3.5
        plain = season_calc(night)
        assert(plain == 3)

        mod2 = season_calc(night, modulo=2)
        assert(mod2 == 1)

        mod3 = season_calc(night, modulo=3)
        assert(mod3 == 0)

        mod3 = season_calc(night, modulo=3, max_season=2)
        assert(mod3 == -1)

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25*2)
        assert(mod3 == 1)

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25*10)
        assert(mod3 == -1)

        mod3 = season_calc(night, modulo=3, offset=-365.25*10)
        assert(mod3 == -1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
