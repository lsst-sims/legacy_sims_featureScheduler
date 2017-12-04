
import numpy as np
import healpy as hp
import lsst.sims.featureScheduler as fs

target_maps = {}
nside = fs.set_default_nside(nside=32)  # Required

target_maps['u'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.31, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)
target_maps['g'] = fs.generate_goal_map(NES_fraction=0.2,
                                        WFD_fraction=0.44, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)
target_maps['r'] = fs.generate_goal_map(NES_fraction=0.46,
                                        WFD_fraction=1.0, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)
target_maps['i'] = fs.generate_goal_map(NES_fraction=0.46,
                                        WFD_fraction=1.0, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)
target_maps['z'] = fs.generate_goal_map(NES_fraction=0.4,
                                        WFD_fraction=0.9, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)
target_maps['y'] = fs.generate_goal_map(NES_fraction=0.,
                                        WFD_fraction=0.9, SCP_fraction=0.15,
                                        GP_fraction=0.15, nside=nside,
                                        generate_id_map=True)

filters = ['u', 'g', 'r', 'i', 'z', 'y']
surveys = []

for filtername in filters:
    bfs = []
    bfs.append(fs.M5_diff_basis_function(filtername=filtername, nside=nside))
    # The position of this bfs must be passed to Driver (see furthermore) for logical reasons
    bfs.append(fs.Target_map_basis_function(filtername=filtername,
                                            target_map=target_maps[filtername][0],
                                            id_map=target_maps[filtername][1],
                                            name_list=target_maps[filtername][2],
                                            out_of_bounds_val=hp.UNSEEN, nside=nside))

    bfs.append(fs.North_south_patch_basis_function(zenith_min_alt=50., nside=nside))
    # bfs.append(fs.Zenith_mask_basis_function(maxAlt=78., penalty=-100, nside=nside))
    bfs.append(fs.Slewtime_basis_function(filtername=filtername, nside=nside))
    bfs.append(fs.Strict_filter_basis_function(filtername=filtername))

    weights = np.array([3.0, 0.2, 1., 3., 3.])
    surveys.append(fs.Greedy_survey_fields(bfs, weights, block_size=1, filtername=filtername, dither=False,
                                           nside=nside, smoothing_kernel=9,
                                           tag_fields=True, tag_map=target_maps[filtername][1]))

scheduler = fs.Core_scheduler(surveys, nside=nside)  # Required
scheduler_visit_counting_bfs = 1  # Required: What is the position of the counting feature on the bfs?
