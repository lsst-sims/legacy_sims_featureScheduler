import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import coordinates as coordinates
from astropy.time import Time
from astropy import units as u
from lsst.sims.utils import Site



class radec2altaz(object):
    def __init__(self, location):
        self.location = location

    def __call__(self, ra, dec, mjd, coords=None):
        aa_frame = coordinates.AltAz(obstime=Time(mjd, format='mjd'), location=self.location)
        if coords is None:
            coords = SkyCoord(ra*u.rad, dec*u.rad)
        aa_coords = coords.transform_to(aa_frame)
        return aa_coords.alt.rad, aa_coords.az.rad


TwoPi = 2.*np.pi
def smallest_signed_angle(a1, a2):
    """Assume angles between 0 and 2 pi
    via https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles"""
    a = (a1 - a2) % TwoPi
    b = (a2 - a1) % TwoPi
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
    """
    def __init__(self, location=None, park_alt=86.5, park_filter='r'):
        self.park_alt_rad = np.radians(park_alt)
        self.park_filter = park_filter
        if location is None:
            self.site = Site('LSST')
            self.location = EarthLocation(lat=self.site.latitude, lon=self.site.longitude,
                                          height=self.site.height)
        # Our RA,Dec to Alt,Az converter
        self.radec2altaz = radec2altaz(self.location)

        self.setup_camera()
        self.setup_dome()
        self.setup_telescope()
        self.setup_optics()

        # Park the telescope
        self.park()

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
        # I'm going to ignore that the old model had the dome altitude at 90.
        self.current_filter = self.park_filter
        self.parked = True
        self.current_coords = None
        self.rotSkyPos = 0.
        self.cumulative_azimuth_rad = 0
        self.cumulative_camera_rad = 0
        # Need to keep tabs on these because we could track a bit
        # So, when we want to compute the cumulative, use these values, not the
        # angles computed from converting self.current_coords to alt,az.
        self.last_az_rad = 0
        self.last_rot_tel_pos_rad = 0

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
                   lax_dome=True):
        """Calculates ``slew'' time to a series of alt/az/filter positions.
        Assumptions (currently):
            assumes rotator is not moved (no rotator included)
            assumes  we never max out cable wrap-around!

        Calculates the ``slew'' time necessary to get from current state
        to alt2/az2/filter2. The time returned is actually the time between
        the end of an exposure at current location and the beginning of an exposure
        at alt2/az2, since it includes readout time in the ``slew'' time.

        Parameters
        ----------
        alt_rad : np.ndarray
            The altitude of the destination pointing in RADIANS.
        az_rad : np.ndarray
            The azimuth of the destination pointing in RADIANS.
        goal_filter : np.ndarray
            The filter to be used in the destination observation.
        lax_dome : boolean, default False
            If True, allow the dome to creep, model a dome slit, and don't
            require the dome to settle in azimuth. If False, adhere to the way
            SOCS calculates slew times (as of June 21 2017).

        Returns
        -------
        np.ndarray
            The number of seconds between the two specified exposures.
        """
        alt_rad, az_rad = self.radec2altaz(ra_rad, dec_rad, mjd)
        current_alt_rad, current_az_rad = self.radec2altaz(0, 0, mjd, coords=self.current_coords)

        deltaAlt = np.abs(alt_rad - current_alt_rad)
        deltaAz = np.abs(az_rad - current_az_rad)
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
                                 (alt_rad < self.telalt_minpos_rad))
        slewTime[outsideLimits] = np.nan
        return slewTime

    def single_slew_time(self, ra_target_rad, dec_target_rad, rotSkyPos, mjd, rotTelPos=None,
                         include_read=True, update=True):
        """Compute the slewtime between current position and a new pointing.
        """

        # Compute the time it takes the dome and telescope to slew to new position
        dome_tel_time = self.slew_times(mjd, ra_target_rad, dec_target_rad)

        # Compute the rotation time as well
        if rotTelPos is None:
            # Compute the taget rotTelPos needed from rotSkyPos
            pass

        # How long does it take to move the rotator
        rotator_time = self._uamSlewTime(deltaRotation, self.telrot_maxspeed_rad, self.telrot_accel_rad)

        slewtime = np.maximum(dome_tel_time, rotator_time)

        # Update the current state of the telescope
        if update:
            self.current_coords = SkyCoord(ra_target_rad*u.rad, dec_target_rad*u.rad)
            self.park = False
            self.rotSkyPos = rotSkyPos
            # Track the cumulative azimuth and camera rotation

        return slewtime

