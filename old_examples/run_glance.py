import lsst.sims.maf.db as db
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.batches import glanceBatch, fOBatch
import argparse


def runGlance(dbfile, outDir='Glance', runName='runname', camera='LSST'):
    conn = db.Database(dbfile, defaultTable='observations')
    colmap = {'ra': 'RA', 'dec': 'dec', 'mjd': 'mjd',
              'exptime': 'exptime', 'visittime': 'exptime', 'alt': 'alt',
              'az': 'az', 'filter': 'filter', 'fiveSigmaDepth': 'fivesigmadepth',
              'night': 'night', 'slewtime': 'slewtime', 'seeingGeom': 'FWHM_geometric',
              'rotSkyPos': 'rotSkyPos', 'raDecDeg': True, 'slewdist': None,
              'note': 'note'}

    #del colmap['note']

    gb = glanceBatch(colmap=colmap, slicer_camera=camera, runName=runName)
    resultsDb = db.ResultsDb(outDir=outDir)

    group = metricBundles.MetricBundleGroup(gb, conn, outDir=outDir, resultsDb=resultsDb)

    group.runAll()
    group.plotAll()

    fb = fOBatch(colmap=colmap, runName=runName)
    group = metricBundles.MetricBundleGroup(fb, conn, outDir=outDir, resultsDb=resultsDb)
    group.runAll()
    group.plotAll()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="run the survey at a glance bundle.")
    parser.add_argument("dbfile", type=str, help="sqlite file")
    parser.add_argument("runName", type=str, default=None, help="run name")
    parser.add_argument("--camera", type=str, default='LSST')
    parser.add_argument("--outDir", type=str, default="Glance")

    args = parser.parse_args()

    runGlance(args.dbfile, runName=args.runName, camera=args.camera, outDir=args.outDir)
