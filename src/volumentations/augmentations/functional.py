import numpy as np
import scipy


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


def noise(points, noise_level=1000):
    bounding_box_diagonal = np.linalg.norm(
        np.max(points[:, :3], axis=0) - np.min(points[:, :3], axis=0)
    )
    return np.random.normal(0, bounding_box_diagonal / noise_level, points.shape)


def elastic_distortion(points, granularity=0.1, magnitude=1.0):
    # from https://github.com/chrischoy/SpatioTemporalSegmentation/blob/4afee296ebe387d9a06fc1b168c4af212a2b4804/lib/transforms.py#L179

    # Create Gaussian blur kernels for each axis
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3

    # Get spatial extents of the points
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    spatial_xyz = points_max - points_min

    # Create Gaussian noise tensor of the size given by granularity
    # - at least 3 for each dimension
    steps_xyz = (spatial_xyz // granularity).astype(int) + 3
    noise = np.random.randn(*steps_xyz, 3).astype(np.float32)

    # Smoothing
    for _ in range(2):
        noise = scipy.ndimage.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate for each spatial dimension
    grid = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(points_min, points_max, steps_xyz)
    ]
    interpolation = scipy.interpolate.RegularGridInterpolator(
        grid, noise, bounds_error=False, fill_value=None
    )

    return interpolation(points) * magnitude
