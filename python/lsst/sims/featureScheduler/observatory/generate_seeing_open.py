import numpy as np
from lsst.sims.ocs.downtime import ScheduledDowntime, UnscheduledDowntime
from lsst.sims.ocs.environment import SeeingModel
import ephem
from lsst.sims.utils import  Site


sec2days = 1./(3600.*24.)
doff = ephem.Date(0)-ephem.Date('1858/11/17')

sun_limit = np.radians(-12.)
mjd0 = 59560.2
survey_length = 15.
mjd_max = mjd0 + 365.25*survey_length


# Now to generate arrays of mjd and zenith seeing so we can do fast lookup if the 
# dome is open and what the seeing is

udt = UnscheduledDowntime()
udt.initialize()
sdt = ScheduledDowntime()
sdt.initialize()

# consolidate a list of nights where things are down.  I need to do weather too I guess.
nights_down = []


#seeing_model = SeeingModel(None)

# I guess I should load up the ephem and make sure the sun isn't up.

sun = ephem.Sun()
site = Site(name='LSST')
obs = ephem.Observer()
obs.lat = site.latitude_rad
obs.lon = site.longitude_rad
obs.elevation = site.height
obs.horizon = 0.

# find the rising and setting sun times

mjd_start = mjd0
mjd_end = mjd_max
step = 0.25
mjds = np.arange(mjd_start, mjd_end+step, step)
setting = mjds*0.
risings = mjds*0.

# Stupid Dublin Julian Date
djds = mjds - doff
sun = ephem.Sun()

for i, (mjd, djd) in enumerate(zip(mjds, djds)):
    sun.compute(djd)
    setting[i] = obs.previous_setting(sun, start=djd, use_center=True)
    risings[i] = obs.next_rising(sun, start=djd, use_center=True)
setting = setting + doff
risings = risings + doff

# zomg, round off crazy floating point precision issues
setting_rough = np.round(setting*100.)
u, indx = np.unique(setting_rough, return_index=True)
setting_sun_mjds = setting[indx]
rising_rough = np.round(risings*100)
u, indx = np.unique(rising_rough, return_index=True)
rising_sun_mjds = risings[indx]

# So, I can loop through the rising,setting values, and append lists of mjds and seeing



#obs.date = mjd-doff
#sun.compute(obs)
#if sun.alt < sun_limit:

