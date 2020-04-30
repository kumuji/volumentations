Examples
========

.. toctree::
   :maxdepth: 2

.. code-block:: python

    import volumentations as V
    import numpy as np

    volume_aug = V.Compose(
        [
            V.Scale3d(scale_limit=[0.1, 0.1, 0.1], bias=[1, 1, 1]),
            V.RotateAroundAxis3d(axis=[0, 0, 1], rotation_limit=np.pi / 6),
            V.RotateAroundAxis3d(axis=[0, 1, 0], rotation_limit=np.pi / 6),
            V.RotateAroundAxis3d(axis=[1, 0, 0], rotation_limit=np.pi / 6),
            V.RandomDropout3d(dropout_ratio=0.2),
        ]
    )
    original_point_cloud = np.empty((1000, 3))
    augmented_point_cloud = volume_aug(points=original_point_cloud)["points"]


For more examples see `repository with examples <https://github.com/kumuji/volumentations/blob/master/examples>`_
