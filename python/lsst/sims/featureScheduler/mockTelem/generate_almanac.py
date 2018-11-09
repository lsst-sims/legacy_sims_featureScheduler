import numpy as np
from astroplan import Observer
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.time import Time
from lsst.sims.utils import Site


# Trying out the astroplan sunrise/set code. 
# conda install -c astropy astroplan
mjd_start=59853.5 -3.*365.25
duration = 25.*365.25
pad_around=40
t_step=0.7

mjds = np.arange(mjd_start-pad_around, duration+mjd_start+pad_around+t_step, t_step)

site = Site('LSST')
observer = Observer(longitude=site.longitude*u.deg, latitude=site.latitude*u.deg,
                   elevation=site.height*u.m, name="LSST")

times = Time(mjds, format='mjd')

sunsets = observer.sun_set_time(times)

sunsets = np.unique(np.round(sunsets.mjd, decimals=4))

names = ['night', 'sunset', 'sun_n12_setting', 'sun_n18_setting', 'sun_n18_rising',
             'sun_n12_rising', 'sunrise', 'moonrise', 'moonset']
types = [int]
types.extend([float]*(len(names)-1))
almanac = np.zeros(sunsets.size, dtype=list(zip(names, types)))
almanac['sunset'] = sunsets


times = Time(sunsets, format='mjd')
almanac['sun_n12_setting'] = observer.twilight_evening_nautical(times).mjd
almanac['sun_n18_setting'] = observer.twilight_evening_astronomical(times).mjd
almanac['sun_n18_rising'] = observer.twilight_morning_astronomical(times).mjd
almanac['sun_n12_rising'] = observer.twilight_evening_astronomical(times).mjd
almanac['sunrise'] = observer.sun_rise_time(times).mjd

almanac['moonset'] = observer.moon_set_time(times).mjd
almanac['moonrise'] = observer.moon_rise_time(times).mjd


np.savez('almanac.npz', almanac=almanac)

