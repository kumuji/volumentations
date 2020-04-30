import random
from copy import deepcopy
from warnings import warn

from volumentations.core.serialization import SerializableMeta
from volumentations.core.six import add_metaclass
from volumentations.core.utils import format_args

__all__ = [
    "to_tuple",
    "BasicTransform",
    "PointCloudsTransform",
    "NoOp",
]


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)


@add_metaclass(SerializableMeta)
class BasicTransform:
    call_backup = None

    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(
                    key in kwargs for key in self.targets_as_params
                ), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(
                    targets_as_params
                )
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname()
                        + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(
        self, params, force_apply=False, **kwargs
    ):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {
                    k: kwargs[k] for k in self.target_dependence.get(key, [])
                }
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag, save_key="replay"):
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(
            name=self.__class__.__name__, args=format_args(state)
        )

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'normals1': 'normals', 'normals2': 'normals'}

        Args:
            additional_targets (dict): keys - new target name, values -
            old target name. ex: {'normals2': 'normals'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class "
            + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self):
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self):
        d = self._to_dict()
        d["id"] = id(self)
        return d


class PointCloudsTransform(BasicTransform):
    """Transform for point clouds."""

    @property
    def targets(self):
        return {
            "points": self.apply,
            "normals": self.apply_to_normals,
            "features": self.apply_to_features,
            "cameras": self.apply_to_camera,
            "bbox": self.apply_to_bboxes,
            "labels": self.apply_to_labels,
        }

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(bbox, **params) for bbox in bboxes]

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError(
            "Method apply_to_bbox is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_cameras(self, cameras, **params):
        return [self.apply_to_bbox(camera, **params) for camera in cameras]

    def apply_to_camera(self, camera, **params):
        raise NotImplementedError(
            "Method apply_to_camera is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_normals(self, normals, **params):
        raise NotImplementedError(
            "Method apply_to_normals is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_features(self, features, **params):
        raise NotImplementedError(
            "Method apply_to_features is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_labels(self, labels, **params):
        raise NotImplementedError(
            "Method apply_to_labels is not implemented in class "
            + self.__class__.__name__
        )


class NoOp(BasicTransform):
    """Does nothing"""

    def apply(self, points, **params):
        return points

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_camera(self, camera, **params):
        return camera

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args_names(self):
        return ()
