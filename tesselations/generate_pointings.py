#!/Users/yoachim/lsst/DarwinX86/miniconda2/3.19.0.lsst4/bin/python
from __future__ import print_function
from lsst.sims.featureScheduler.thomson import even_points_xyz, xyz2thetaphi
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
    x0 = even_points_xyz(npts)

    x0 = x0.reshape(3, x0.size/3)
    x = x0[0, :]
    y = x0[1, :]
    z = x0[2, :]

    theta, phi = xyz2thetaphi(x, y, z)

    np.savez('npts_%i.npz' % npts, theta=theta, phi=phi)
