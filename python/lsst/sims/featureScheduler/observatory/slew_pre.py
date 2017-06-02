import numpy as np
import os
from lsst.utils import getPackageDir
from scipy.interpolate import RectBivariateSpline


__all__ = ['Slewtime_pre']


class Slewtime_pre(object):
    """Calculate the slewtime from a pre-computed grid
    """
    def __init__(self):
        path = getPackageDir('sims_featureScheduler')
        datafile = os.path.join(path, 'python/lsst/sims/featureScheduler/observatory/pre_slewtimes.npz')
        data = np.load(datafile)
        alt_array = data['alt_array'].copy()
        az_array = data['az_array'].copy()
        altitudes = np.radians(data['altitudes'].copy())
        azimuths = np.radians(data['azimuths'].copy())
        data.close()
        # Generate interpolation objects
        self.alt_interpolator = RectBivariateSpline(altitudes, altitudes, alt_array)
        self.az_interpolator = RectBivariateSpline(azimuths, azimuths, az_array)

    def __call__(self, alt1, az1, alt2, az2):
        """
        Parameters
        ----------
        alt1 : float
            Altitude of current pointing (radians)
        az1 : float
            Azimuth of current pointing (radians)
        alt2 : np.array
            Array of possible altitudes to slew to (radians)
        az2 : np.array
            Array of possible azimuths to slew to (radians)

        Returns
        -------
        Slewtimes in seconds. Includes a minimum of 2 seconds because it assumes there was a
        readout started right before slew. Currently does not worry about camera rotation.
        """

        result = np.empty((2, np.size(alt2)))
        result[0, :] = self.alt_interpolator(alt1, alt2, grid=False)
        result[1, :] = self.az_interpolator(az1, az2, grid=False)
        result = np.max(result, axis=0)
        return result

