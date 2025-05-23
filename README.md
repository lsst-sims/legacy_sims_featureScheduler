Archived -- please see [rubin_scheduler](github.com/lsst/rubin_scheduler) instead.

# sims_featureScheduler
Telescope scheduler that uses basis functions computed from features to optimize observing strategy


# Installation instructions

Can be installed with the LSST sims: https://confluence.lsstcorp.org/display/SIM/Catalogs+and+MAF#/

# Output Schema

The `sim_runner` function converts the simulation output to try and match previous output schemas


All values are for the center of the field of view (e.g., airmass, altitude, etc)


 |  Column Name  |  Units  |  Description  | 
 |  ---  |  ---  |  ---  | 
 | airmass |   unitless  |  airmass of the observation (center of the field) | 
 | altitude |  degrees  |  Altitude of the observation | 
 | azimuth |  degrees  |  Azimuth of the observation  | 
 | block_id |  int  |  Identification ID of the block (used by some survey objects) | 
 | cloud |  fraction  |  what fraction of the sky is cloudy | 
 | fieldDec |  degrees  |  Declination of the observation | 
 | fieldId |  int  |  deprecated, should all be 0 or -1. | 
 | fieldRA |  degrees  |  Right Ascension of the observation | 
 | filter |  string  |  The filter that was loaded for the observation, one of u,g,r,i,z,y | 
 | fiveSigmaDepth |  magnitudes  |  The magnitude of an isolated point source detected at the 5-sigma level | 
 | flush_by_mjd |  days  |  The modified Julian date the observation would have been flushed from the queue at | 
 | moonAlt |  degrees  |  Altitude of the moon | 
 | moonAz |  degrees  |  Azimuth of the moon | 
 | moonDec |  degrees  |  Declination of the moon | 
 | moonDistance |  degrees  |  Angular distance between the observation and the moon | 
 | moonPhase |  percent (0-100)  |  The phase of the moon (probably the same as illumination fraction) | 
 | moonRA |  degrees  |  Right Ascension of the moon | 
 | night |  days  |  The night of the survey (starting at 1) | 
 | note |  string  |  Note added by the scheduler, often which survey object generated the observation | 
 | numExposures |  int  |  Number of exposures in the visit | 
 | observationId |  int  |  Unique observation ID | 
 | observationStartLST |  degrees  |  the Local Sidereal Time at the start of the observation | 
 | observationStartMJD |  days  |  Modified Julian Date at the start of the observation | 
 | paraAngle |  degrees  |  Paralactic angle of the observation | 
 | proposalId |  int  |  deprecated | 
 | rotSkyPos |  degrees  |  The orientation of the sky in the focal plane measured as the angle between North on the sky and the "up" direction in the focal plane. | 
 | rotTelPos |  degrees | The physical angle of the rotator with respect to the mount. rotSkyPos = rotTelPos - ParallacticAngle | 
 | seeingFwhm500 |  arcseconds  |  The full-width at half maximum of the PSF at 500 nm. (XXX-unsure if this is at zenith or at the pointing) | 
 | seeingFwhmEff |  arcseconds  |  "Effective" full-width at half maximum, typically ~15% larger than FWHMgeom. Use FWHMeff to calculate SNR for point sources, using FWHMeff as the FWHM of a single Gaussian describing the PSF. | 
 | seeingFwhmGeom |  arcseconds  |  "Geometrical" full-width at half maximum. The actual width at half the maximum brightness. Use FWHMgeom to represent the FWHM of a double-Gaussian representing the physical width of a PSF. | 
 | skyBrightness |  mag arcsec^-2  |  the brightness of the sky (in the given filter) for the observation | 
 | slewDistance |  degrees  |  distance the telescope slewed to the observation | 
 | slewTime |  seconds  |  The time it took to slew to the observation. Includes any filter change time and any readout time. | 
 | solarElong |  degrees  |  Solar elongation or the angular distance between the field center and the sun (0 - 180 deg). | 
 | sunAlt |  degrees  |  Altitude of the sun | 
 | sunAz |  degrees  |  Azimuth of the sun | 
 | sunDec |  degrees  |  declination of the sun | 
 | sunRA |  degrees  |  RA of the sun | 
 | visitExposureTime |  seconds  |  Total exposure time of the visit | 
 | visitTime |  seconds  |  Total time of the visit (could be larger than `visitExposureTime` if the visit had multiple exposures with readout between them) | 
 | cummTelAz |  degrees  |  The cumulative azimuth rotation of the telescope mount, should be +/- 270 degrees due to cable wrap limits.  | 
