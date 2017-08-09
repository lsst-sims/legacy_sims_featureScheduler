import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.bundles import glanceBundle
import argparse


def runGlance(dbfile, outDir='Glance', runName='runname'):
    conn = db.SimpleDatabase(dbfile, defaultTable='SummaryAllProps',
                             defaultdbTables = {'SummaryAllProps': ['SummaryAllProps', 'observationId']})
    colmap = {'ra': 'RA', 'dec': 'dec', 'mjd': 'mjd',
                  'exptime': 'exptime', 'visittime': 'exptime', 'alt': 'alt',
                  'az': 'az', 'filter': 'filter', 'fiveSigmaDepth': 'fivesigmadepth',
                  'night': 'night', 'slewtime': 'slewtime', 'seeingGeom': 'FWHM_geometric'}

    gb = glanceBundle(colmap_dict=colmap)
    resultsDb = db.ResultsDb(outDir=outDir)

    group = metricBundles.MetricBundleGroup(gb, conn, outDir=outDir, resultsDb=resultsDb,
                                            runName=runName)

    group.runAll()
    group.plotAll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run the survey at a glance bundle.")
    parser.add_argument("dbfile", type=str, help="sqlite file")
    parser.add_argument("runName", type=str, default=None, help="run name")

    args = parser.parse_args()

    runGlance(args.dbfile, runName=args.runName)
