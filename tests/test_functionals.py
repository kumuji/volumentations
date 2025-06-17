import numpy as np
import pytest
import volumentations.augmentations.functional as F


@pytest.mark.parametrize(
    ["points", "axis", "angle", "expected_points"],
    [
        (
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            [1, 0, 0],
            np.pi / 2,
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
        )
    ],
)
def test_rotate(points, angle, axis, expected_points):
    # rotation around axis x
    processed_points = F.rotate_around_axis(points, axis=axis, angle=angle)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "min_max"],
    [
        (
            np.array([[100, 0, 0], [-100, 0, 0], [0, 0, 0], [0, 1, 100], [0, 100, 1]]),
            np.array([0, 0, 0]),
            (-10, 10, -10, 10, -10, 10),
        )
    ],
)
def test_crop(points, expected_points, min_max):
    processed_points = points[
        F.crop(
            points,
            x_min=min_max[0],
            x_max=min_max[1],
            y_min=min_max[2],
            y_max=min_max[3],
            z_min=min_max[4],
            z_max=min_max[5],
        )
    ]
    assert np.allclose(expected_points, processed_points)


def test_crop_should_fail(points):
    try:
        F.crop(
            points,
            x_min=10,
            x_max=0,
            y_min=0,
            y_max=10,
            z_min=0,
            z_max=10,
        )
        assert False, "Crop input limits check didn't work"
    except ValueError:
        pass


@pytest.mark.parametrize(
    ["points", "expected_points"],
    [
        (
            np.array([[2, 2, 2], [0, 0, 0]], dtype=float),
            np.array([[1, 1, 1], [-1, -1, -1]], dtype=float),
        )
    ],
)
def test_center(points, expected_points):
    processed_points = F.center(points)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "offset"],
    [
        (
            np.array([[2, 2, 2], [0, 0, 0]], dtype=float),
            np.array([[102, 142, 52], [100, 140, 50]], dtype=float),
            np.array([100, 140, 50]),
        )
    ],
)
def test_move(points, expected_points, offset):
    processed_points = F.move(points, offset=offset)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "scale_factor"],
    [
        (
            np.array([[2, 1, 1], [0, 0, 0]], dtype=float),
            np.array([[4, 1, 1], [0, 0, 0]], dtype=float),
            np.array([2, 1, 1], dtype=float),
        )
    ],
)
def test_scale(points, expected_points, scale_factor):
    processed_points = F.scale(points, scale_factor)
    assert np.allclose(expected_points, processed_points)


@pytest.mark.parametrize(
    ["points", "expected_points", "axis"],
    [
        (
            np.array([[1, 1, 1], [0, 0, 0]], dtype=float),
            np.array([[0, 1, 1], [1, 0, 0]], dtype=float),
            np.array([1, 0, 0]),
        )
    ],
)
def test_flip(points, expected_points, axis):
    processed_points = F.flip_coordinates(points, axis)
    assert np.allclose(expected_points, processed_points)
