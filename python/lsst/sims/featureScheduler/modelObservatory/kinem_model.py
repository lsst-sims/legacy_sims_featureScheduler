import numpy as np
from astropy.coordinates import EarthLocation
from lsst.sims.utils import Site, calcLmstLast
import healpy as hp
import matplotlib.pylab as plt
from lsst.sims.featureScheduler.utils import approx_altaz2pa

__all__ = ["Kinem_model"]
TwoPi = 2.*np.pi


# Snagged from lsst.sims.utils for now to add in parallactic angle. Might want to update back there
def _approx_RaDec2AltAz(ra, dec, lat, lon, mjd, lmst=None, return_pa=True):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore aberration, precession, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    if lmst is None:
        lmst, last = calcLmstLast(mjd, lon)
    lmst = lmst/12.*np.pi  # convert to rad
    ha = lmst-ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec*sinlat+np.cos(dec)*coslat*np.cos(ha)
    sinalt = np.clip(sinalt, -1, 1)
    alt = np.arcsin(sinalt)
    cosaz = (sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat)
    cosaz = np.clip(cosaz, -1, 1)
    az = np.arccos(cosaz)
    if np.size(ha) < 2:
        if np.sin(ha) > 0:
            az = 2.*np.pi-az
    else:
        signflip = np.where(np.sin(ha) > 0)
        az[signflip] = 2.*np.pi-az[signflip]
    if return_pa:
        pa = approx_altaz2pa(alt, az, lat)
        return alt, az, pa
    return alt, az


class radec2altazpa(object):
    """Class to make it easy to swap in different alt/az conversion if wanted
    """
    def __init__(self, location):
        self.location = location

    def __call__(self, ra, dec, mjd):
        alt, az, pa = _approx_RaDec2AltAz(ra, dec, self.location.lat.rad, self.location.lon.rad, mjd)
        return alt, az, pa


def _getRotSkyPos(paRad, rotTelRad):
    """
    Paramteres
    ----------
    paRad : float or array
        The parallactic angle
    """
    return (rotTelRad - paRad) % TwoPi


def _getRotTelPos(paRad, rotSkyRad):
    """Make it run from -180 to 180
    """
    result = (rotSkyRad + paRad) % TwoPi
    return result


def smallest_signed_angle(a1, a2):
    """
    via https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles"""
    x = a1 % TwoPi
    y = a2 % TwoPi
    a = (x - y) % TwoPi
    b = (y - x) % TwoPi
    return -a if a < b else b


class Kinem_model(object):
    """
    A Kinematic model of the telescope.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation object
        The location of the telescope. If None, defaults to lsst.sims.utils.Site info
    park_alt : float (86.5)
        The altitude the telescope gets parked at (degrees)
    park_filter : str ('r')
        The filter that gets loaded when the telescope is parked

    Note there are additional parameters in the methods setup_camera, setup_dome, setup_telescope,
    and setup_optics. Just breaking it up a bit to make it more readable.
    """
    def __init__(self, location=None, park_alt=86.5, park_az=0., park_filter='r'):
        self.park_alt_rad = np.radians(park_alt)
        self.park_az_rad = np.radians(park_az)
        self.park_filter = park_filter
        if location is None:
            self.site = Site('LSST')
            self.location = EarthLocation(lat=self.site.latitude, lon=self.site.longitude,
                                          height=self.site.height)
        # Our RA,Dec to Alt,Az converter
        self.radec2altaz = radec2altazpa(self.location)

        self.setup_camera()
        self.setup_dome()
        self.setup_telescope()
        self.setup_optics()

        # Park the telescope
        self.park()

    def mount_filters(self, filter_list):
        """Change which filters are mounted
        """
        self.mounted_filters = filter_list

    def setup_camera(self, readtime=2., shuttertime=1., filter_changetime=120., fov=3.5,
                     rotator_min=-90, rotator_max=90, maxspeed=3.5, accel=1.0, decel=1.0):
        """
        Parameters
        ----------
        readtime : float (2)
            The readout time of the CCDs (seconds)
        shuttertime : float (1.)
            The time it takes the shutter to go from closed to fully open (seconds)
        filter_changetime : float (120)
            The time it takes to change filters (seconds)
        fov : float (3.5)
            The camera field of view (degrees)
        rotator_min : float (-90)
            The minimum angle the camera rotator can move to (degrees)
        maxspeed : float (3.5)
            The maximum speed of the rotator (degrees/s)
        accel : float (1.0)
            The acceleration of the rotator (degrees/s^2)
        """
        self.readtime = readtime
        self.shuttertime = shuttertime
        self.filter_changetime = filter_changetime
        self.camera_fov = np.radians(fov)

        self.telrot_minpos_rad = np.radians(rotator_min)
        self.telrot_maxpos_rad = np.radians(rotator_max)
        self.telrot_maxspeed_rad = np.radians(maxspeed)
        self.telrot_accel_rad = np.radians(accel)
        self.telrot_decel_rad = np.radians(decel)
        self.mounted_filters = ['u', 'g', 'r', 'i', 'y']

    def setup_dome(self, altitude_maxspeed=1.75, altitude_accel=0.875, altitude_decel=0.875,
                   altitude_freerange=0., azimuth_maxspeed=1.5, azimuth_accel=0.75,
                   azimuth_decel=0.75, azimuth_freerange=4.0, settle_time=1.0):
        """input in degrees, degees/second, degrees/second**2, and seconds.
        Freerange is the range in which there is zero delay.
        """
        self.domalt_maxspeed_rad = np.radians(altitude_maxspeed)
        self.domalt_accel_rad = np.radians(altitude_accel)
        self.domalt_decel_rad = np.radians(altitude_decel)
        self.domalt_free_range = np.radians(altitude_freerange)
        self.domaz_maxspeed_rad = np.radians(azimuth_maxspeed)
        self.domaz_accel_rad = np.radians(azimuth_accel)
        self.domaz_decel_rad = np.radians(azimuth_decel)
        self.domaz_free_range = np.radians(azimuth_freerange)
        self.domaz_settletime = settle_time

    def setup_telescope(self, altitude_minpos=20.0, altitude_maxpos=86.5,
                        azimuth_minpos=-270.0, azimuth_maxpos=270.0, altitude_maxspeed=3.5,
                        altitude_accel=3.5, altitude_decel=3.5, azimuth_maxspeed=7.0,
                        azimuth_accel=7.0, azimuth_decel=7.0, settle_time=3.0):
        """input in degrees, degees/second, degrees/second**2, and seconds.
        """
        self.telalt_minpos_rad = np.radians(altitude_minpos)
        self.telalt_maxpos_rad = np.radians(altitude_maxpos)
        self.telaz_minpos_rad = np.radians(azimuth_minpos)
        self.telaz_maxpos_rad = np.radians(azimuth_maxpos)
        self.telalt_maxspeed_rad = np.radians(altitude_maxspeed)
        self.telalt_accel_rad = np.radians(altitude_accel)
        self.telalt_decel_rad = np.radians(altitude_decel)
        self.telaz_maxspeed_rad = np.radians(azimuth_maxspeed)
        self.telaz_accel_rad = np.radians(azimuth_accel)
        self.telaz_decel_rad = np.radians(azimuth_decel)
        self.mount_settletime = settle_time

    def setup_optics(self, ol_slope=1.0/3.5, cl_delay=[0.0, 36.], cl_altlimit=[0.0, 9.0, 90.0]):
        """
        Parameters
        ----------
        ol_slope : float
            seconds/degree in altitude slew.
        cl_delay : list ([0.0, 36])
            The delays for closed optics loops (seconds)
        cl_altlimit : list ([0.0, 9.0, 90.0])
            The altitude limits (degrees) for performing closed optice loops. Should be one element longer than cl_delay.
        """

        self.optics_ol_slope = ol_slope/np.radians(1.)  # ah, 1./np.radians(1)=np.pi/180
        self.optics_cl_delay = cl_delay
        self.optics_cl_altlimit = np.radians(cl_altlimit)

    def park(self):
        """Put the telescope in the park position.
        """
        # I'm going to ignore that the old model had the dome altitude at 90 and telescope altitude 86 for park.
        # We should usually be dome az limited anyway, so this should be a negligible approximation.

        self.parked = True

        # We have no current position we are tracking
        self.current_RA_rad = None
        self.current_dec_rad = None
        self.current_rotSkyPos_rad = None
        self.current_filter = self.park_filter
        self.cumulative_azimuth_rad = 0

        # The last position we were at (or the current if we are parked)
        self.last_az_rad = self.park_az_rad
        self.last_alt_rad = self.park_alt_rad
        self.last_rot_tel_pos_rad = 0

    def current_alt_az(self, mjd):
        """return the current alt az position that we have tracked to.
        """
        if self.parked:
            return self.last_alt_rad, self.last_az_rad, self.last_rot_tel_pos_rad
        else:
            alt_rad, az_rad, pa = self.radec2altaz(self.current_RA_rad, self.current_dec_rad, mjd)
            rotTelPos = _getRotTelPos(pa, self.last_rot_tel_pos_rad)
            return alt_rad, az_rad, rotTelPos

    def _uamSlewTime(self, distance, vmax, accel):
        """Compute slew time delay assuming uniform acceleration (for any component).
        If you accelerate uniformly to vmax, then slow down uniformly to zero, distance traveled is
        d  = vmax**2 / accel
        To travel distance d while accelerating/decelerating at rate a, time required is t = 2 * sqrt(d / a)
        If hit vmax, then time to acceleration to/from vmax is 2*vmax/a and distance in those
        steps is vmax**2/a. The remaining distance is (d - vmax^2/a) and time needed is (d - vmax^2/a)/vmax

        This method accepts arrays of distance, and assumes acceleration == deceleration.

        Parameters
        ----------
        distance : numpy.ndarray
            Distances to travel. Must be positive value.
        vmax : float
            Max velocity
        accel : float
            Acceleration (and deceleration)

        Returns
        -------
        numpy.ndarray
        """
        dm = vmax**2 / accel
        slewTime = np.where(distance < dm, 2 * np.sqrt(distance / accel),
                            2 * vmax / accel + (distance - dm) / vmax)
        return slewTime

    def slew_times(self, ra_rad, dec_rad, mjd, rotSkyPos=None, rotTelPos=None, filtername='r',
                   lax_dome=True, alt_rad=None, az_rad=None, starting_alt_rad=None, starting_az_rad=None,
                   starting_rotTelPos_rad=None, update_tracking=False, include_readtime=True):
        """Calculates ``slew'' time to a series of alt/az/filter positions from the current
        position (stored internally).
        Assumptions (currently):
            assumes  we never max out cable wrap-around!--For now--XXX to update
            Assumes we have been tracking on ra,dec,rotSkyPos position.
            Ignores the motion of the sky while we are slewing.
            No checks for if we have tracked beyond limits.

        Calculates the ``slew'' time necessary to get from current state
        to alt2/az2/filter2. The time returned is actually the time between
        the end of an exposure at current location and the beginning of an exposure
        at alt2/az2, since it includes readout time in the ``slew'' time.

        Parameters
        ----------
        ra_rad : np.ndarray
            The RA(s) of the location(s) we wish to slew to (radians)
        dec_rad : np.ndarray
            The declination(s) of the location(s) we wish to slew to (radians)
        mjd : float
            The current moodified julian date (days)
        rotSkyPos : np.ndarray
            The desired rotSkyPos(s) (radians). Angle between up on the chip and North. Note,
            it is possible to set a rotSkyPos outside the allowed camera rotator range, in which case
            the slewtime will be np.inf. If both rotSkyPos and rotTelPos are set, rotTelPos will be used.
        rotTelPos : np.ndarray
            The desired rotTelPos(s) (radians).
        filtername : str
            The filter(s) of the desired observations. Set to None to compute only telescope and dome motion times.
        alt_rad : np.ndarray
            The altitude(s) of the destination pointing(s) (radians). Will override ra_rad,dec_rad if provided.
        az_rad : np.ndarray
            The azimuth(s) of the destination pointing(s) (radians). Will override ra_rad,dec_rad if provided.
        lax_dome : boolean, default False
            If True, allow the dome to creep, model a dome slit, and don't
            require the dome to settle in azimuth. If False, adhere to the way
            SOCS calculates slew times (as of June 21 2017).
        starting_alt_rad : float (None)
            The starting altitude for the slew (radians). If None, will use internally stored last pointing.
        starting_az_rad : float (None)
            The starting azimuth for the slew (radians). If None, will use internally stored last pointing.
        starting_rotTelPos_rad : float (None)
            The starting camera rotation for the slew (radians). If None, will use internally stored last pointing.
        update_tracking : bool (False)
            If True, update the internal attributes to say we are tracking the specified RA,Dec,RotSkyPos position.
        include_readtime : bool (True)
            Assume the camera must be read before opening the shutter, and include that readtime in the returned slewtime.
            Readtime will never be inclded if the telescope was parked before the slew.

        Returns
        -------
        np.ndarray
            The number of seconds between the two specified exposures.
        """
        if filtername not in self.mounted_filters:
            return np.nan

        # Don't trust folks to do pa calculation correctly, if both rotations set, rotSkyPos wins
        if (rotTelPos is not None) & (rotSkyPos is not None):
            if np.isfinite(rotTelPos):
                rotSkyPos = None
            else:
                rotTelPos = None

        # alt,az not provided, calculate from RA,Dec
        if alt_rad is None:
            alt_rad, az_rad, pa = self.radec2altaz(ra_rad, dec_rad, mjd)
        if starting_alt_rad is None:
            if self.parked:
                starting_alt_rad = self.park_alt_rad
                starting_az_rad = self.park_az_rad
            else:
                starting_alt_rad, starting_az_rad, starting_pa = self.radec2altaz(self.current_RA_rad,
                                                                                  self.current_dec_rad, mjd)

        deltaAlt = np.abs(alt_rad - starting_alt_rad)
        deltaAz = np.abs(az_rad - starting_az_rad)
        deltaAz = np.minimum(deltaAz, np.abs(deltaAz - 2 * np.pi))

        # Calculate how long the telescope will take to slew to this position.
        telAltSlewTime = self._uamSlewTime(deltaAlt, self.telalt_maxspeed_rad,
                                           self.telalt_accel_rad)
        telAzSlewTime = self._uamSlewTime(deltaAz, self.telaz_maxspeed_rad,
                                          self.telaz_accel_rad)
        totTelTime = np.maximum(telAltSlewTime, telAzSlewTime)
        # Time for open loop optics correction
        olTime = deltaAlt / self.optics_ol_slope
        totTelTime += olTime
        # Add time for telescope settle.
        # XXX--note, this means we're going to have a settle time even for very small slews (like even a dither)
        settleAndOL = np.where(totTelTime > 0)
        totTelTime[settleAndOL] += np.maximum(0, self.mount_settletime - olTime[settleAndOL])
        # And readout puts a floor on tel time
        if include_readtime:
            totTelTime = np.maximum(self.readtime, totTelTime)

        # now compute dome slew time
        if lax_dome:
            # model dome creep, dome slit, and no azimuth settle
            # if we can fit both exposures in the dome slit, do so
            sameDome = np.where(deltaAlt ** 2 + deltaAz ** 2 < self.camera_fov ** 2)

            # else, we take the minimum time from two options:
            # 1. assume we line up alt in the center of the dome slit so we
            #    minimize distance we have to travel in azimuth.
            # 2. line up az in the center of the slit
            # also assume:
            # * that we start out going maxspeed for both alt and az
            # * that we only just barely have to get the new field in the
            #   dome slit in one direction, but that we have to center the
            #   field in the other (which depends which of the two options used)
            # * that we don't have to slow down until after the shutter
            #   starts opening
            domDeltaAlt = deltaAlt
            # on each side, we can start out with the dome shifted away from
            # the center of the field by an amount domSlitRadius - fovRadius
            domSlitDiam = self.camera_fov / 2.0
            domDeltaAz = deltaAz - 2 * (domSlitDiam / 2 - self.camera_fov / 2)
            domAltSlewTime = domDeltaAlt / self.domalt_maxspeed_rad
            domAzSlewTime = domDeltaAz / self.domaz_maxspeed_rad
            totDomTime1 = np.maximum(domAltSlewTime, domAzSlewTime)

            domDeltaAlt = deltaAlt - 2 * (domSlitDiam / 2 - self.camera_fov / 2)
            domDeltaAz = deltaAz
            domAltSlewTime = domDeltaAlt / self.domalt_maxspeed_rad
            domAzSlewTime = domDeltaAz / self.domaz_maxspeed_rad
            totDomTime2 = np.maximum(domAltSlewTime, domAzSlewTime)

            totDomTime = np.minimum(totDomTime1, totDomTime2)
            totDomTime[sameDome] = 0

        else:
            # the above models a dome slit and dome creep. However, it appears that
            # SOCS requires the dome to slew exactly to each field and settle in az
            domAltSlewTime = self._uamSlewTime(deltaAlt, self.domalt_maxspeed_rad,
                                               self.domalt_accel_rad)
            domAzSlewTime = self._uamSlewTime(deltaAz, self.domaz_maxspeed_rad,
                                              self.domaz_accel_rad)
            # Dome takes 1 second to settle in az
            domAzSlewTime = np.where(domAzSlewTime > 0,
                                     domAzSlewTime + self.domaz_settletime,
                                     domAzSlewTime)
            totDomTime = np.maximum(domAltSlewTime, domAzSlewTime)
        # Find the max of the above for slew time.
        slewTime = np.maximum(totTelTime, totDomTime)
        # include filter change time if necessary
        filterChange = np.where(filtername != self.current_filter)
        slewTime[filterChange] = np.maximum(slewTime[filterChange],
                                            self.filter_changetime)
        # Add closed loop optics correction
        # Find the limit where we must add the delay
        cl_limit = self.optics_cl_altlimit[1]
        cl_delay = self.optics_cl_delay[1]
        closeLoop = np.where(deltaAlt >= cl_limit)
        slewTime[closeLoop] += cl_delay

        # Mask min/max altitude limits so slewtime = np.nan
        outsideLimits = np.where((alt_rad > self.telalt_maxpos_rad) |
                                 (alt_rad < self.telalt_minpos_rad))[0]
        slewTime[outsideLimits] = np.nan

        # If we want to include the camera rotation time
        if (rotSkyPos is not None) | (rotTelPos is not None):
            if rotTelPos is None:
                rotTelPos = _getRotTelPos(pa, rotSkyPos)
            if rotSkyPos is None:
                rotSkyPos = _getRotSkyPos(pa, rotTelPos)
            # If the new rotation angle would move us out of the limits, return nan
            rotTelPos_ranged = rotTelPos+0
            over = np.where(rotTelPos > np.pi)[0]
            rotTelPos_ranged[over] -= TwoPi
            if (rotTelPos_ranged < self.telrot_minpos_rad) | (rotTelPos_ranged > self.telrot_maxpos_rad):
                return np.nan
            # If there was no kwarg for starting rotator position
            if starting_rotTelPos_rad is None:
                # If there is no current rotSkyPos, we were parked
                if self.current_rotSkyPos_rad is None:
                    current_rotTelPos = self.last_rot_tel_pos_rad
                else:
                    # We have been tracking, so rotTelPos needs to be updated
                    current_rotTelPos = _getRotTelPos(pa, self.current_rotSkyPos_rad)
            else:
                # kwarg overrides if it was supplied
                current_rotTelPos = starting_rotTelPos_rad
            deltaRotation = np.abs(smallest_signed_angle(current_rotTelPos, rotTelPos))
            rotator_time = self._uamSlewTime(deltaRotation, self.telrot_maxspeed_rad, self.telrot_accel_rad)
            slewTime = np.maximum(slewTime, rotator_time)

        # Update the internal attributes to note that we are now pointing and tracking
        # at the requested RA,Dec,rotSkyPos
        if update_tracking:
            self.current_RA_rad = ra_rad
            self.current_dec_rad = dec_rad
            self.current_rotSkyPos_rad = rotSkyPos
            self.parked = False
            self.last_rot_tel_pos_rad = rotTelPos
            self.last_az_rad = az_rad
            self.last_alt_rad = alt_rad
            self.last_pa_rad = pa
            # Track the cumulative azimuth
            # XXX--change this to the slew distance that was used (large or small depending)
            self.cumulative_azimuth_rad += smallest_signed_angle(self.cumulative_azimuth_rad, az_rad)
            self.current_filter = filtername
            self.last_mjd = mjd

        return slewTime

    def visit_time(self, observation):
        # How long does it take to make an observation. Assume final read can be done during next slew.
        visit_time = observation['exptime'] + \
            observation['nexp'] * self.shuttertime + \
            max(observation['nexp'] - 1, 0) * self.readtime
        return visit_time

    def observe(self, observation, mjd, rotTelPos=None):
        """observe a target, and return the slewtime and visit time for the action

        If slew is not allowed, returns np.nan and does not update state.
        """
        slewtime = self.slew_times(observation['RA'], observation['dec'],
                                   mjd, rotSkyPos=observation['rotSkyPos'],
                                   rotTelPos=rotTelPos,
                                   filtername=observation['filter'], update_tracking=True)
        visit_time = self.visit_time(observation)
        return slewtime, visit_time