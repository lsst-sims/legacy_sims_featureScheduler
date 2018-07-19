import sqlite3
from lsst.sims.featureScheduler.coldstart.database import *
from lsst.sims.featureScheduler.utils import empty_observation
import numpy as np
import math
import pickle



def get_observation_history(filePath):
    '''queries the opsim database and returns a list of past observations for cold start'''

    db = database()
    c = db.connect(filePath)

    c.execute(("SELECT ra, dec, observationStartMJD, visitExposureTime, " 
    "filter, angle, numExposures, airmass, seeingFwhmEff, seeingFwhmGeom, "
    "skyBrightness, night, slewTime, fiveSigmaDepth, " 
    "altitude, azimuth, cloud, moonAlt, sunAlt, note, Field_fieldId, proposal_propId FROM ObsHistory "
    "JOIN SlewHistory ON (ObsHistory.observationId = SlewHistory.ObsHistory_observationId) JOIN "
    "ObsProposalHistory ON (ObsHistory.observationId = ObsProposalHistory.ObsHistory_observationId)"))
    
    fields_in_fbsObs = \
    ('RA', 'dec', 'mjd', 'exptime', 'filter', 'rotSkyPos', 'nexp', 'airmass', 
    'FWHMeff', 'FWHM_geometric', 'skybrightness', 'night', 'slewtime', 
    'fivesigmadepth', 'alt', 'az', 'clouds', 'moonAlt', 'sunAlt', 'note', 
    'field_id','survey_id')

    #some fields are stored in the database as degrees, but need to be converted to radians.
    convert_to_rad = ('RA', 'dec', 'alt', 'az', 'moonAlt', 'sunAlt', 'rotSkyPos')

    obslist = []
    for obs in c.fetchall():
        o = empty_observation()
        for i in range(len(fields_in_fbsObs)):
            if fields_in_fbsObs[i] in convert_to_rad:
                o[fields_in_fbsObs[i]] = math.radians(obs[i])
            else: 
                o[fields_in_fbsObs[i]] = obs[i]
        obslist.append(o)
    c.close()
    return obslist

if __name__ == '__main__':
    o = get_observation_history()

    f = open("first847fbs","rb")
    obs_from_file = pickle.load(f)

    for i in range(800):
        print(o[i][0])
        print(obs_from_file[i][0])
        print()

