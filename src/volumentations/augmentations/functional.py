import numpy as np


def scale(points, scale_factor=(1, 1, 1)):
    transformation_matrix = np.eye(3)
    np.fill_diagonal(transformation_matrix, scale_factor)
    points[:, :3] = np.dot(points[:, :3], transformation_matrix)
    return points


def rotate_around_axis(points, axis, angle, center_point=None):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by angle in radians.
    https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    if center_point is None:
        center_point = points[:, :3].mean(axis=0).astype(points[:, :3].dtype)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )
    points[:, :3] = points[:, :3] - center_point
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    points[:, :3] = points[:, :3] + center_point
    return points


def crop(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(
                x_min=x_min,
                x_max=x_max,
                y_min=y_min,
                y_max=y_max,
                z_min=z_min,
                z_max=z_max,
            )
        )
    inds = np.all(
        [
            (points[:, 0] >= x_min),
            (points[:, 0] < x_max),
            (points[:, 1] >= y_min),
            (points[:, 1] < y_max),
            (points[:, 2] >= z_min),
            (points[:, 2] < z_max),
        ],
        axis=0,
    )
    return inds


def center(points, origin=(0, 0, 0)):
    points[:, :3] -= origin + points[:, :3].mean(axis=0)
    return points


def move(points, offset=(0, 0, 0)):
    points[:, :3] = points[:, :3] + offset
    return points


def flip_coordinates(points, axis):
    axis = np.argmax(axis)
    coord_max = np.max(points[:, axis])
    points[:, axis] = coord_max - points[:, axis]
    return points


# import scipy

# def elastic_distortion(pointcloud, granularity, magnitude):
#     """Apply elastic distortion on sparse coordinate space.

#     pointcloud: numpy array of (number of points, at least 3 spatial dims)
#     granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
#     magnitude: noise multiplier
#   """
#     blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
#     blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
#     blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
#     coords = pointcloud[:, :3]
#     coords_min = coords.min(0)

#     # Create Gaussian noise tensor of the size given by granularity.
#     noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
#     noise = np.random.randn(*noise_dim, 3).astype(np.float32)

#     # Smoothing.
#     for _ in range(2):
#         noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
#         noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
#         noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

#     # Trilinear interpolate noise filters for each spatial dimensions.
#     ax = [
#         np.linspace(d_min, d_max, d)
#         for d_min, d_max, d in zip(
#             coords_min - granularity,
#             coords_min + granularity * (noise_dim - 2),
#             noise_dim,
#         )
#     ]
#     interp = scipy.interpolate.RegularGridInterpolator(
#         ax, noise, bounds_error=0, fill_value=0
#     )
#     pointcloud[:, :3] = coords + interp(coords) * magnitude
#     return pointcloud
