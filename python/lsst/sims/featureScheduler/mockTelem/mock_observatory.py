import numpy as np
from lsst.sims.utils import _hpid2RaDec, _raDec2Hpid, Site, calcLmstLast, _approx_RaDec2AltAz
import lsst.sims.skybrightness_pre as sb
import healpy as hp
from lsst.sims.utils import m5_flat_sed, _angularSeparation
from datetime import datetime
from lsst.sims.featureScheduler import version
from lsst.sims.downtimeModel import ScheduledDowntime, UnscheduledDowntime
from lsst.sims.seeingModel import SeeingSim
from lsst.sims.cloudModel import CloudModel
from lsst.sims.featureScheduler.features import Conditions
from lsst.sims.featureScheduler.utils import set_default_nside
from astropy.coordinates import get_sun, get_moon, EarthLocation, AltAz
from astropy.time import Time


__all__ = ['Mock_observatory']




def generate_sunsets(mjd_start, duration=12.):
    """Generate the sunset and twilight times for a range of dates
    
    Parameters
    ----------
    mjd_start : float
        The starting mjd
    duration : float (12.)
        How long to compute times for (years)
    """

    # Should also do moon-rise, set times.

    # end result, I want an array so that given an MJD I can:
    # look up if it's day, night, or twililight
    # What night number it is
    # When the next moon rise/set is.


    # Let's find the nights first, find the times where the sun crosses the meridian.
    site = Site('LSST')
    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    # go on 1/10th of a day steps
    t_step = 0.1
    t = Time(np.arange(duration*365.25*t_step, t_step)+mjd_start, format='mjd', location=location)
    sun = get_sun(t)
    aa = AltAz(location=location, obstime=t)
    sun_aa = sun.transform_to(aa)



class filter_swap_scheduler(object):
    """A simple way to schedule what filter to load
    """
    def __init__(self):
        pass

    def __call__(self, conditions):
        # Just based on lunar illumination, decide if some different filter should be loaded.
        pass



class Mock_observatory(object):
    """A class to generate a realistic telemetry stream for the scheduler
    """

    def __init__(self, nside=None, mjd_start=59853.5):
        """
        Parameters
        ----------
        nside : int (None)
            The healpix nside resolution
        """

        if nside is None:
            nside = set_default_nside()
        self.nside = nside

        self.mjd = mjd_start

        self.conditions = Conditions(nside=self.nside)

        # Load up all the models we need
        # 



    def return_conditions(self):
        """
        Returns
        -------
        lsst.sims.featureScheduler.features.conditions object
        """

        
        # 



        return self.conditions

    def set_mjd(self):
        pass

    def observe(self, observation):
        """Try to make an observation
        """

        return observation


