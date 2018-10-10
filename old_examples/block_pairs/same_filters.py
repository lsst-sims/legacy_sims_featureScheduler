import numpy as np
import lsst.sims.featureScheduler as fs
from lsst.sims.speedObservatory import Speed_observatory
import matplotlib.pylab as plt
import healpy as hp
from lsst.sims.utils import _hpid2RaDec
import time

t0 = time.time()

survey_length = 366 # days
nside = fs.set_default_nside(nside=32)


# Let's set up some alt_az maps
# XXX--this is an initial guess at what reasoable blocks might look like! 
alt_az_blockmaps = []
hpids = np.arange(hp.nside2npix(nside))
az, alt = _hpid2RaDec(nside, hpids)

# XXX--seem like reasonable numbers, but who knows if they are hte best!
alt_low_limit = np.radians(20.)
alt_high_limit = np.radians(82.)
az_half_width = np.radians(15.)

blank_map = hpids *0.
good = np.where( ((az > 2.*np.pi-az_half_width) | (az < az_half_width)) &
                (alt > alt_low_limit) & (alt < alt_high_limit))
new_map = blank_map +0
new_map[good] = 1
alt_az_blockmaps.append(new_map)

good = np.where(((az > np.pi-az_half_width) & (az < np.pi+az_half_width)) &
                (alt > alt_low_limit) & (alt < alt_high_limit))
new_map = blank_map +0
new_map[good] = 1
alt_az_blockmaps.append(new_map)

# let's make some finer grained steps. Bring up the lower limit a bit.
alt_high_limit = np.radians(75.)
alt_low_limit = np.radians(35.)

steps = np.arange(20, 330, 20)
for step in steps:
    good = np.where((az > np.radians(step)) & (az < np.radians(step)+az_half_width*2) &
                   (alt > alt_low_limit) & (alt < alt_high_limit))
    new_map = blank_map + 0
    new_map[good] = 1
    alt_az_blockmaps.append(new_map)


# Define what we want the final visit ratio map to look like
years = np.round(survey_length/365.25)
# get rid of silly northern strip.
target_map = fs.standard_goals(nside=nside)

# List to hold all the surveys (for easy plotting later)
surveys = []

# Set up observations to be taken in blocks
filter1s = ['u', 'g', 'r', 'i', 'z', 'y']
filter2s = [None, 'g', 'r', 'i', None, None]
pair_surveys = []
for filtername, filtername2 in zip(filter1s, filter2s):
    bfs = []
    bfs.append(fs.M5_diff_basis_function(filtername=filtername, nside=nside))
    if filtername2 is not None:
        bfs.append(fs.M5_diff_basis_function(filtername=filtername2, nside=nside))
    bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                            target_map=target_map[filtername],
                                            out_of_bounds_val=hp.UNSEEN, nside=nside))
    if filtername2 is not None:
        bfs.append(fs.Target_map_basis_function(filtername=filtername2,
                                            target_map=target_map[filtername2],
                                            out_of_bounds_val=hp.UNSEEN, nside=nside))
    bfs.append(fs.Slewtime_basis_function(filtername=filtername, nside=nside))
    # XXX--put a huge boost on staying in the filter. Until I can tier surveys as list-of-lists
    weights = np.array([3.0, 3.0, 0.3, 0.3, 3.])
    if filtername2 is None:
        # Need to scale weights up so filter balancing still works.
        weights = np.array([6.0, 0.6, 3.])
    # XXX-
    # This is where we could add a look-ahead basis function to include m5_diff in the future.
    # Actually, having a near-future m5 would also help prevent switching to u or g right at twilight?
    # Maybe just need a "filter future" basis function?
    if filtername2 is None:
        survey_name = 'block, %s' % filtername
    else:
        survey_name = 'block, %s%s' % (filtername, filtername2)
    surveys.append(fs.Block_survey(bfs, weights, filtername=filtername, filter2=filtername2,
                                   dither=True, nside=nside, ignore_obs='DD',
                                   alt_az_masks=alt_az_blockmaps,
                                   survey_note=survey_name))
    pair_surveys.append(surveys[-1])


# Let's set up some standard surveys as well to fill in the gaps. This is my old silly masked version.
# It would be good to put in Tiago's verion and lift nearly all the masking. That way this can also
# chase sucker holes.
filters = ['u', 'g', 'r', 'i', 'z', 'y']
greedy_surveys = []
for filtername in filters:
    bfs = []
    bfs.append(fs.M5_diff_basis_function(filtername=filtername, nside=nside))
    bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                            target_map=target_map[filtername],
                                            out_of_bounds_val=hp.UNSEEN, nside=nside))

    bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50., nside=nside))
    bfs.append(fs.Slewtime_basis_function(filtername=filtername, nside=nside))
    bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

    weights = np.array([3.0, 0.3, 1., 3., 3.])
    # Might want to try ignoring DD observations here, so the DD area gets covered normally--DONE
    surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername,
                                           dither=True, nside=nside, ignore_obs='DD'))
    greedy_surveys.append(surveys[-1])

# Set up the DD surveys
#dd_surveys = []
dd_surveys = fs.generate_dd_surveys()
surveys.extend(dd_surveys)


survey_list_o_lists = [pair_surveys, greedy_surveys]
#survey_list_o_lists = [dd_surveys, pair_surveys, greedy_surveys]

# Debug to stop at a spot if needed
n_visit_limit = None

# put in as list-of-lists so pairs get evaluated first.
scheduler = fs.Core_scheduler(survey_list_o_lists, nside=nside)
observatory = Speed_observatory(nside=nside, quickTest=False)
observatory, scheduler, observations = fs.sim_runner(observatory, scheduler,
                                                     survey_length=survey_length,
                                                     filename='same_pairs_%iyrs.db' % years,
                                                     delete_past=True, n_visit_limit=n_visit_limit)

t1 = time.time()
delta_t = t1-t0
print('ran in %.1f min = %.1f hours' % (delta_t/60., delta_t/3600.))