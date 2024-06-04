import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


# cosmo functions

def cosmo_to_astropy(params):
    """Converts a list of cosmological parameters into an astropy cosmology
    object. Note, ignores s8 and n_s parameters, which are not used in astropy.

    Args:
        params (list): [Omega_m, Omega_b, h, n_s, s8]
    """

    # check if params is a list
    try:
        params = list(params)
    except TypeError:
        return params
    return FlatLambdaCDM(H0=params[2]*100, Om0=params[0], Ob0=params[1])


def xyz_to_sky(pos, vel, cosmo):
    """Converts cartesian coordinates to sky coordinates (ra, dec, z).
    Inspired by nbodykit.transform.CartesianToSky.

    Args:
        pos (array): (N, 3) array of comoving positions in Mpc/h.
        vel (array): (N, 3) array of comoving velocities in km/s.
        cosmo (array): Cosmological parameters
            [Omega_m, Omega_b, h, n_s, sigma8].
    """
    cosmo = cosmo_to_astropy(cosmo)

    pos /= cosmo.h  # convert from Mpc/h to Mpc
    pos *= u.Mpc  # label as Mpc
    vel *= u.km / u.s  # label as km/s

    # get ra, dec
    coord_cart = SkyCoord(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        representation_type='cartesian')
    coord_sphe = coord_cart.represent_as('spherical')
    ra = coord_sphe.lon.to(u.deg)
    dec = coord_sphe.lat.to(u.deg)

    # get redshift
    R = np.linalg.norm(pos, axis=-1)

    def z_from_comoving_distance(d):
        zgrid = np.logspace(-8, 1.5, 2048)
        zgrid = np.concatenate([[0.], zgrid])
        dgrid = cosmo.comoving_distance(zgrid)
        return interp1d(dgrid, zgrid)(d)

    # Convert comoving distance to redshift
    z = z_from_comoving_distance(R)

    vpec = (pos*vel).sum(axis=-1) / R
    z += vpec / c.to(u.km/u.s)*(1+z)

    return np.array([ra, dec, z]).T


def sky_to_xyz(rdz, cosmo):
    """Converts sky coordinates (ra, dec, z) to cartesian coordinates.

    Args:
        rdz (array): (N, 3) array of sky coordinates (ra, dec, z).
        cosmo (array): Cosmological parameters
            [Omega_m, Omega_b, h, n_s, sigma8].
    """
    cosmo = cosmo_to_astropy(cosmo)

    ra, dec, z = rdz.T
    pos = SkyCoord(ra=ra*u.deg, dec=dec*u.deg,
                   distance=cosmo.comoving_distance(z))
    pos = pos.cartesian.xyz
    pos *= cosmo.h  # convert from Mpc to Mpc/h

    return pos.value.T


def rotate_to_z(rdz, cosmo):
    """Returns a Rotation object which rotates a sightline (ra, dec, z) to
    the z-axis.

    Args:
        rdz (array): (3,) array of sky coordinates (ra, dec, z).
        cosmo (array): Cosmological parameters
            [Omega_m, Omega_b, h, n_s, sigma8].

    Returns:
        rot (Rotation): Rotation object which rotates the sightline to z-axis.
        irot (Rotation): Inverse rotation object.
    """

    # calculate comoving vector
    xyz = sky_to_xyz(rdz, cosmo)
    mvec = xyz / np.linalg.norm(xyz)

    # use a nearby point in +RA to affix x-axis
    rdz1 = rdz + [0.001, 0, 0]  # add 0.001
    xyz1 = sky_to_xyz(rdz1, cosmo)

    # calculate x-axis vector (to be rotated later to x-axis)
    xvec = xyz1 - xyz
    xvec /= np.linalg.norm(xvec)

    # rotate xyz to z
    rotz_axis = np.cross(mvec, [0, 0, 1])
    rotz_axis /= np.linalg.norm(rotz_axis)
    rotz_angle = np.arccos(np.dot(mvec, [0, 0, 1]))
    rotz = R.from_rotvec(rotz_angle*rotz_axis)

    # rotate xvec to x-axis
    xvec = rotz.apply(xvec)
    rotx_angle = -np.arctan2(xvec[1], xvec[0])
    rotx = R.from_rotvec(rotx_angle*np.array([0, 0, 1]))

    # combine rotations and measure inverse
    rot = rotx*rotz
    irot = rot.inv()

    return rot, irot
