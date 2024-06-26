import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from copy import deepcopy


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


def random_rotate_translate(xyz, L, vel=None, seed=0):
    """Randomly rotate and translate a cube of points.

    Rotations are fixed to [0, 90, 180, 270] degrees on each axis,
    to satisfy periodic boundary conditions.

    Args:
    - xyz (np.ndarray): (N, 3) array of positions in the cube.
    - L (float): side length of the cube.
    - vel (np.ndarray, optional): (N, 3) array of velocities. 
    - seed (int): random seed for reproducibility. If 0, no transformation
        is applied.
    """
    assert np.all((xyz >= 0) & (xyz <= L)), "xyz must be in [0, L]"
    xyz, vel = map(deepcopy, [xyz, vel])

    if seed == 0:
        offset = np.zeros(3)
        rotation = R.identity()
    else:
        np.random.seed(seed)
        offset = np.random.rand(3)*L
        rotation = R.from_euler(
            'xyz', np.random.choice([0, 90, 180, 270], 3),
            degrees=True)

    # Rotate
    xyz -= L/2
    xyz = rotation.apply(xyz)
    xyz += L/2
    vel = rotation.apply(vel) if vel is not None else None

    # Translate
    xyz += offset
    xyz %= L

    return xyz, vel


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
    rdz = np.asarray(rdz)

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


# Everything below is hardcoded, from pre-computation

def get_cuboid_lattice_vectors():
    """Returns the lattice vectors for a cuboid which fits any quadrant."""
    u1 = [-1, -1, -1]
    u2 = [-1, -1, 0]
    u3 = [-1, 0, -1]
    return u1, u2, u3


def get_median_rdz(i):
    """Returns the median ra, dec, z for each quadrant.

    Args:
        i (int): Quadrant index (0, 1, 2, 3).

    Returns:
        median_rdz (array): Median ra, dec, z for the quadrant.
    """

    precomputed_medians = [
        [158.74303878318312, 17.031270102890435, 0.026811392977833748],
        [153.8592001343185, 44.158932963996975, 0.026938676834106445],
        [216.71722613494188, 16.329543211502916, 0.027733768336474895],
        [222.9051448119211, 40.377727202747, 0.02919937949627638]
    ]
    return np.array(precomputed_medians[i])


def get_mean_xyz(i):
    """Returns the mean comoving position for each quadrant.

    Args:
        i (int): Quadrant index (0, 1, 2, 3).

    Returns:
        mean_xyz (array): Mean x, y, z comoving position for the quadrant.
    """

    precomputed_means = [
        [-7.423230881631724, -1.7948550618949621, 63.590013506439035],
        [-1.3064032063107227, 9.825968802370923, 63.63458562404868],
        [5.699150317635436, -0.32144799152813697, 63.56326396869353],
        [0.30577017327591705, 13.15838995467428, 63.762795256487166]
    ]
    return np.array(precomputed_means[i])


def label_quadrants(ra, dec):
    """Labels each ra, dec pair with a quadrant index.

    Args:
        ra (array): Array of ra values.
        dec (array): Array of dec values.

    Returns:
        labels (array): Array of quadrant labels.
    """
    ramid = 187
    decmid = 30

    labels = np.zeros(ra.shape, dtype=int)
    labels[(ra < ramid) & (dec < decmid)] = 0
    labels[(ra < ramid) & (dec > decmid)] = 1
    labels[(ra > ramid) & (dec < decmid)] = 2
    labels[(ra > ramid) & (dec > decmid)] = 3
    return labels
