#!/usr/bin/env python

import logging
import sys

from lsst.ts.scheduler import Main
from lsst.ts.scheduler.setup import configure_logging, create_parser, generate_logfile
from lsst.sims.featureScheduler.driver import Driver

def main(args):
    logfilename = generate_logfile()
    configure_logging(args, logfilename, args.log_port)

    logger = logging.getLogger("scheduler")
    logger.info("logfile=%s" % logfilename)

    scheduler = Main(args, Driver())
    scheduler.sal_init()
    with open('.scheduler_{}'.format(args.log_port), 'w'):
        pass
    scheduler.run()

    sys.exit(0)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.profile:
        import cProfile
        from datetime import datetime
        cProfile.run("main(args)",
                     "scheduler_prof_{}.dat".format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    else:
        main(args)
