import numpy as np
import healpy as hp
from lsst.ts.observatory.model import ObservatoryModel
from lsst.ts.dateloc import DateProfile, ObservatoryLocation
import copy
from lsst.sims.ocs.observatory import MainObservatory
from lsst.sims.ocs.configuration import ObservingSite, Observatory
# Generate a few matrices to make it easy to generate slewtime maps


class target_spoof(object):
    """Make a dummy class that can pass as a  SALPY_scheduler.targetC
    """
    def __init__(self, targetId=None, fieldId=None, filtername=None, ra=0., dec=0.,
                 angle=0., num_exposures=2, exposure_times=15.):
        self.targetId = targetId
        self.fieldId = fieldId
        self.filter = filtername
        self.ra = ra
        self.dec = dec
        self.angle = angle
        self.num_exposures = num_exposures
        self.exposure_times = [exposure_times]*num_exposures
        self.ra_rad = np.radians(ra)
        self.dec_rad = np.radians(dec)
        self.ang_rad = np.radians(angle)


def slewtime(model, alt1=None, alt2=None, az1=None, az2=None, rot1=None, rot2=None):
    # park the observatory, then go from alt1,az1 to alt2,az2
    if alt1 is None:
        alt1 = model.current_state.alt
    if alt2 is None:
        alt2 = model.current_state.alt
    if az1 is None:
        az1 = model.current_state.az
    if az2 is None:
        az2 = model.current_state.az
    if rot1 is None:
        rot1 = model.current_state.rot
    if rot2 is None:
        rot2 = model.current_state.rot

    model.park()
    model.slew_altaz(0., np.radians(alt1), np.radians(az1), np.radians(rot1), model.current_state.filter)
    initial_slew_state = copy.deepcopy(model.current_state)

    model.slew_altaz(model.current_state.time, np.radians(alt2), np.radians(az2), np.radians(rot2),
                     model.current_state.filter)

    final_slew_state = copy.deepcopy(model.current_state)
    slew_time = (final_slew_state.time - initial_slew_state.time, "seconds")

    return slew_time

observatory_location = ObservatoryLocation()

target = target_spoof(ra=50., dec=-20.)
# Easy to just make an observatory and pull out the configured model
mo = MainObservatory(ObservingSite())
mo.configure(Observatory())
model = mo.model


delta = 0.5

azimuths = np.arange(0, 360+delta, delta)
altitudes = np.arange(20., 90.+delta, delta)

alt_array = np.zeros((altitudes.size, altitudes.size), dtype=float)
az_array = np.zeros((azimuths.size, azimuths.size), dtype=float)

for i in range(azimuths.size):
    for j in range(azimuths.size):
        st = slewtime(model, az1=azimuths[i], az2=azimuths[j])
        az_array[i, j] += st[0]

for i in range(altitudes.size):
    for j in range(altitudes.size):
        st = slewtime(model, alt1=altitudes[i], alt2=altitudes[j])
        alt_array[i, j] += st[0]

np.savez('pre_slewtimes.npz', alt_array=alt_array, az_array=az_array, azimuths=azimuths, altitudes=altitudes)
