import warnings
import sys
import numpy as np
from lsst.sims.featureScheduler.utils import run_info_table, schema_converter
from lsst.sims.featureScheduler.schedulers import simple_filter_sched
import time
import sqlite3
import pandas as pd

__all__ = ['sim_runner']


def sim_runner(observatory, scheduler, filter_scheduler=None, mjd_start=None, survey_length=3.,
               filename=None, delete_past=True, n_visit_limit=None, step_none=15., verbose=True,
               extra_info=None, event_table=None):
    """
    run a simulation

    Parameters
    ----------
    survey_length : float (3.)
        The length of the survey ot run (days)
    step_none : float (15)
        The amount of time to advance if the scheduler fails to return a target (minutes).
    extra_info : dict (None)
        If present, dict gets added onto the information from the observatory model.
    event_table : np.array (None)
        Any ToO events that were included in the simulation
    """

    if extra_info is None:
        extra_info = {}

    t0 = time.time()

    if filter_scheduler is None:
        filter_scheduler = simple_filter_sched()

    if mjd_start is None:
        mjd = observatory.mjd
        mjd_start = mjd + 0
    else:
        mjd = mjd_start + 0
        observatory.mjd = mjd

    end_mjd = mjd + survey_length
    observations = []
    mjd_track = mjd + 0
    step = 1./24.
    step_none = step_none/60./24.  # to days
    mjd_run = end_mjd-mjd_start
    nskip = 0
    new_night = False

    while mjd < end_mjd:
        if not scheduler._check_queue_mjd_only(observatory.mjd):
            scheduler.update_conditions(observatory.return_conditions())
        desired_obs = scheduler.request_observation(mjd=observatory.mjd)
        if desired_obs is None:
            # No observation. Just step into the future and try again.
            warnings.warn('No observation. Step into the future and trying again.')
            observatory.mjd = observatory.mjd + step_none
            scheduler.update_conditions(observatory.return_conditions())
            nskip += 1
            continue
        completed_obs, new_night = observatory.observe(desired_obs)
        if completed_obs is not None:
            scheduler.add_observation(completed_obs[0])
            observations.append(completed_obs)
            filter_scheduler.add_observation(completed_obs[0])
        else:
            scheduler.flush_queue()
        if new_night:
            # find out what filters we want mounted
            conditions = observatory.return_conditions()
            filters_needed = filter_scheduler(conditions)
            swap_out = np.setdiff1d(conditions.mounted_filters, filters_needed)
            for filtername in swap_out:
                # ugh, "swap_filter" means "unmount filter"
                observatory.observatory.swap_filter(filtername)

        mjd = observatory.mjd
        if verbose:
            if (mjd-mjd_track) > step:
                progress = float(mjd-mjd_start)/mjd_run*100
                text = "\rprogress = %.2f%%" % progress
                sys.stdout.write(text)
                sys.stdout.flush()
                mjd_track = mjd+0
        if n_visit_limit is not None:
            if len(observations) == n_visit_limit:
                break
        # XXX--handy place to interupt and debug
        # if len(observations) > 3:
        #    import pdb ; pdb.set_trace()
    runtime = time.time() - t0
    print('Skipped %i observations' % nskip)
    print('Flushed %i observations from queue for being stale' % scheduler.flushed)
    print('Completed %i observations' % len(observations))
    print('ran in %i min = %.1f hours' % (runtime/60., runtime/3600.))
    print('Writing results to ', filename)
    observations = np.array(observations)[:, 0]
    if filename is not None:
        info = run_info_table(observatory, extra_info=extra_info)
        converter = schema_converter()
        converter.obs2opsim(observations, filename=filename, info=info, delete_past=delete_past)
    if event_table is not None:
        df = pd.DataFrame(event_table)
        con = sqlite3.connect(filename)
        df.to_sql('events', con)
        con.close()
    return observatory, scheduler, observations
