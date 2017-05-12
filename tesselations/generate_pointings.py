#!/Users/yoachim/lsst/DarwinX86/miniconda2/3.19.0.lsst4/bin/python
from __future__ import print_function
from lsst.sims.featureScheduler.thomson import even_points, thetaphi2xyz
import numpy as np
import argparse

# Let's make some pointings
# Run in parallel with:
# cat commandFile.txt | xargs -n15 -I'{}' -P3 bash -c '{}'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve the Thomson problem')
    parser.add_argument('npts', type=int, help='Number of points to put on a sphere')
    args = parser.parse_args()
    npts = args.npts
    print('Solving Thomson problem for %i points' % npts)
    theta, phi = even_points(npts)

    np.savez('npts_%i.npz' % npts, theta=theta, phi=phi)
