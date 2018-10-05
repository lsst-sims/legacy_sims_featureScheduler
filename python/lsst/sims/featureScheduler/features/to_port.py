
#XXX---previously made condition features that need to be migrated to the Conditions object.


class Time_to_set(BaseConditionsFeature):
    """Map of how much time until things set.

    Warning, using very fast alt/az transformation ignoring refraction, aberration, etc.
    """
    def __init__(self, nside=default_nside, alt_min=20.):
        """
        Parameters
        ----------
        alt_min : float
            The minimum altitude one can point the telescope (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.min_alt = np.radians(alt_min)

        self.sin_dec = np.sin(self.dec)
        self.cos_dec = np.cos(self.dec)

        site = Site('LSST')
        self.sin_lat = np.sin(site.latitude_rad)
        self.cos_lat = np.cos(site.latitude_rad)
        self.lon = site.longitude_rad

        # Compute hour angle when field hits the alt_min
        ha_alt_min = -np.arccos((np.sin(self.min_alt) -
                                 self.sin_dec*self.sin_lat)/(self.cos_dec*self.cos_lat))
        self.ha_alt_min = ha_alt_min
        lmst_alt_min = ha_alt_min + self.ra
        lmst_alt_min[np.where(lmst_alt_min < 0)] += 2.*np.pi
        self.lmst_min = lmst_alt_min

        self.nans = np.isnan(self.lmst_min)

    def update_conditions(self, conditions):
        """feature = time to set in hours
        """
        lmst = conditions['lmst'] * np.pi/12.

        rad_to_limit = self.lmst_min - lmst
        rad_to_limit[np.where(rad_to_limit < 0)] += 2.*np.pi

        self.feature = rad_to_limit
        self.feature *= 12/np.pi * 365.24/366.24
        self.feature[self.nans] = hp.UNSEEN


class Time_to_alt_limit(BaseConditionsFeature):
    """Map of how much time until things set.

    Warning, using very fast alt/az transformation ignoring refraction, aberration, etc.
    """
    def __init__(self, nside=default_nside, alt_max=86.5):
        """
        Parameters
        ----------
        alt_max : float
            The maximum altitude one can point the telescope (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.max_alt = np.radians(alt_max)

        self.sin_dec = np.sin(self.dec)
        self.cos_dec = np.cos(self.dec)

        site = Site('LSST')
        self.sin_lat = np.sin(site.latitude_rad)
        self.cos_lat = np.cos(site.latitude_rad)
        self.lon = site.longitude_rad

        # compute the hour angle when a point hits the alt_max
        cos_ha = (np.sin(self.max_alt) - self.sin_dec*self.sin_lat)/(self.cos_dec * self.cos_lat)
        self.lmst_max = np.arccos(cos_ha) + self.ra
        self.nans = np.isnan(self.lmst_max)

    def update_conditions(self, conditions):
        """feature = time to set in hours
        """
        lmst = conditions['lmst'] * np.pi/12.

        rad_to_limit = self.lmst_max - lmst
        rad_to_limit[np.where(rad_to_limit < 0)] += 2.*np.pi

        self.feature = rad_to_limit
        self.feature *= 12/np.pi * 365.24/366.24
        self.feature[self.nans] = hp.UNSEEN



class Time_observable_in_night(BaseConditionsFeature):
    """
    For every healpixel, calculate the time left observable in the night
    """
    def __init__(self, nside=default_nside, max_airmass=2.5, polar_limit=-80.):
        """
        Parameters
        ----------
        max_airmass : float (2.5)
            The maximum airmass to consider a point visible
        polar_limit : float (-80.)
            Consider anything below dec polar_limit to always be visible. (degrees)
        """
        if nside is None:
            nside = utils.set_default_nside()

        # most fields should have a min and max lmst where they are less than max_airmass

        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))

        alt_limit = np.pi/2. - np.arccos(1./max_airmass)
        site = Site('LSST')
        lat = site.latitude_rad
        self.lon = site.longitude_rad

        sinalt = np.sin(alt_limit)
        sindec = np.sin(self.dec)
        sinlat = np.sin(lat)
        cosdec = np.cos(self.dec)
        coslat = np.cos(lat)

        cosha = (sinalt - sindec*sinlat)/(cosdec*coslat)
        # Here's the hour angle (plus or minus) for each healpixel
        self.ha_limit = np.arccos(cosha)*12/np.pi

        # Consider some regions circumpolar
        self.ha_limit[np.where(self.dec < np.radians(polar_limit))] = 12.

        self.polar_limit = polar_limit

        self.feature = self.ra * 0.

    def update_conditions(self, conditions):
        """
        self.feature : healpy map
            The hours remaining for a field to be visible in the night before the
            next twilight starts.
        """
        # reset the feature value
        self.feature *= 0.

        lmst = conditions['lmst']
        # Map of the current HA, in hours
        current_ha = lmst - (self.ra*12./np.pi)
        current_ha[np.where(current_ha < 0.)] += 24
        # now to convert to -12 to 12
        over = np.where(current_ha > 12.)
        current_ha[over] = current_ha[over] - 24.

        # in hours
        time_to_twilight = (conditions['next_twilight_start'] - conditions['mjd']) * 24.

        # Check if still in twilight.
        if np.abs(conditions['mjd'] -
                  conditions['next_twilight_end']) < np.abs(conditions['mjd'] -
                                                            conditions['last_twilight_end']):
            time_left_twilight = (conditions['next_twilight_end'] - conditions['mjd']) * 24
        else:
            time_left_twilight = 0.

        # Convert from sidereal hours to regular hours. Thanks wikipedia!
        side2solar = 365.24/366.24

        # time until next setting
        self.feature = (self.ha_limit - current_ha) * side2solar

        # Crop off if hits twilight first
        self.feature[np.where(self.feature > time_to_twilight)] = time_to_twilight

        # If it needs to rise, subtract that time off
        good = np.where(current_ha < -self.ha_limit)
        time_to_rise = (-self.ha_limit[good] - current_ha[good]) * side2solar
        self.feature[good] = self.feature[good]-time_to_rise

        # If we are still in twilight, subtract that time off
        self.feature -= time_left_twilight

        # Crop off things that won't rise in time.
        self.feature[np.where(self.feature < 0)] = 0.

        # Set the polar region to be the time to twilight
        self.feature[np.where(self.dec <
                              np.radians(self.polar_limit))] = time_to_twilight - time_left_twilight

