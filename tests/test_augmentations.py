import numpy as np
import pytest
from volumentations import (
    Center3d,
    Crop3d,
    Move3d,
    NoOp,
    RandomDropout3d,
    RotateAroundAxis3d,
    Scale3d,
)


@pytest.mark.parametrize(
    ["augmentation_cls", "params"],
    [
        [Scale3d, {"scale_limit": (0, 0, 0)}],
        [RotateAroundAxis3d, {"rotation_limit": 0}],
        [Move3d, {}],
        [RandomDropout3d, {"dropout_ratio": 0.0}],
        [Crop3d, {}],
        [NoOp, {}],
    ],
)
def test_augmentations_wont_change_input(
    augmentation_cls, params, points, features, labels, normals, bboxes, cameras
):
    points_copy = points.copy()
    features_copy = features.copy()
    labels_copy = labels.copy()
    normals_copy = normals.copy()
    aug = augmentation_cls(p=1, **params)
    data = aug(
        points=points,
        features=features,
        labels=labels,
        normals=normals,
    )
    np.testing.assert_allclose(data["points"], points_copy)
    np.testing.assert_allclose(data["features"], features_copy)
    np.testing.assert_allclose(data["labels"], labels_copy)
    np.testing.assert_allclose(data["normals"], normals_copy)


@pytest.mark.parametrize(
    "points", [np.ones((100, 3)), np.zeros((100, 3)), np.full((100, 3), 42.0)]
)
def test_center(points, features, labels, normals, bboxes, cameras):
    points_copy = np.zeros((100, 3))
    features_copy = features.copy()
    labels_copy = labels.copy()
    normals_copy = normals.copy()
    aug = Center3d(p=1)
    data = aug(
        points=points,
        features=features,
        labels=labels,
        normals=normals,
    )
    np.testing.assert_allclose(data["points"], points_copy)
    np.testing.assert_allclose(data["features"], features_copy)
    np.testing.assert_allclose(data["labels"], labels_copy)
    np.testing.assert_allclose(data["normals"], normals_copy)
