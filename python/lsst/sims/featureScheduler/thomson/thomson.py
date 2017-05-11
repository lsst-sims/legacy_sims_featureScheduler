import numpy as np
from scipy.optimize import minimize
from lsst.sims.utils import _angularSeparation

__all__ = ['thetaphi2xyz', 'even_points', 'elec_potential', 'ang_potential']


def thetaphi2xyz(theta, phi):
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def elec_potential(x0):
    """
    Compute the potential energy for electrons on a sphere

    Parameters
    ----------
    x0 : array
       First half of x0 or theta values, secnd half phi

    Returns
    -------
    Potential energy
    """

    theta = x0[0:x0.size/2]
    phi = x0[x0.size/2:]

    x, y, z = thetaphi2xyz(theta, phi)
    # Distance squared
    dsq = 0.

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices]-coord_j[indices])**2
        dsq += d

    U = np.sum(1./np.sqrt(dsq))
    return U


def ang_potential(x0):
    """
    If distance is compted along sphere rather than 3-space.
    # XXX--this doesn't seem to work yet.
    """
    theta = x0[0:x0.size/2]
    phi = x0[x0.size/2:]

    indices = np.triu_indices(theta.size, k=1)

    theta_i = np.tile(theta, (theta.size, 1))
    theta_j = theta_i.T
    phi_i = np.tile(phi, (phi.size, 1))
    phi_j = phi_i.T
    d = _angularSeparation(theta_i[indices], phi_i[indices], theta_j[indices], phi_j[indices])
    U = np.sum(1./d)
    return U


def fib_sphere_grid(npoints):
    """
    Use a Fibonacci spiral to distribute points uniformly on a sphere.

    based on https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid_points.py

    Returns theta and phi in radians
    """

    phi = (1.0 + np.sqrt(5.0)) / 2.0

    i = np.arange(npoints, dtype=float)
    i2 = 2*i - (npoints-1)
    theta = (2.0*np.pi * i2/phi) % (2.*np.pi)
    sphi = i2/npoints
    phi = np.arccos(sphi)
    return theta, phi


def even_points(npts, use_fib_init=True, method='CG', potential_func=elec_potential):
    """
    Distribute npts over a sphere and minimize their potential, making them
    "evenly" distributed

    Starting with the Fibonacci spiral speeds things up by ~factor of 2.
    """

    if use_fib_init:
        # Start with fibonacci spiral guess
        theta, phi = fib_sphere_grid(npts)
    else:
        # Random on a sphere
        theta = np.random.rand(npts)*np.pi*2.
        phi = np.arccos(2.*np.random.rand(npts)-1.)

    x = np.concatenate((theta, phi))
    # XXX--need to check if this is the best minimizer
    min_fit = minimize(potential_func, x, method='CG')

    x = min_fit.x
    theta = x[0:x.size/2]
    phi = x[x.size/2:]
    # Looks like I get the same energy values as https://en.wikipedia.org/wiki/Thomson_problem
    return theta, phi

