import functools
from typing import Callable, Union, Type, Mapping, Any

from ray.rllib.utils.annotations import override
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.models.specs.specs_base import TensorSpecs


_MISSING_KEYS_FROM_SPEC = (
    "The data dict does not match the model specs. Keys {} are "
    "in the data dict but not on the given spec dict, and exact_match is set to True."
)
_MISSING_KEYS_FROM_DATA = (
    "The data dict does not match the model specs. Keys {} are "
    "in the spec dict but not on the data dict."
)
_TYPE_MISMATCH = (
    "The data does not match the spec. The data element "
    "{} has type {} (expected type {})."
)

SPEC_LEAF_TYPE = Union[Type, TensorSpecs]
DATA_TYPE = Union[NestedDict[Any], Mapping[str, Any]]

IS_NOT_PROPERTY = "Spec {} must be a property of the class {}."


class ModelSpecDict(NestedDict[SPEC_LEAF_TYPE]):
    """A NestedDict containing `TensorSpecs` and `Types`.

    It can be used to validate an incoming data against a nested dictionary of specs.

    Examples:

        Basic validation:
        -----------------
        >>> spec_dict = ModelSpecDict({
        ...     "obs": {
        ...         "arm":      TensorSpecs("b, d_a", d_a=64),
        ...         "gripper":  TensorSpecs("b, d_g", d_g=12)
        ...     },
        ...     "action": TensorSpecs("b, d_a", h=12),
        ...     "action_dist": torch.distributions.Categorical
        ... })

        >>> spec_dict.validate({
        ...     "obs": {
        ...         "arm":      torch.randn(32, 64),
        ...         "gripper":  torch.randn(32, 12)
        ...     },
        ...     "action": torch.randn(32, 12),
        ...     "action_dist": torch.distributions.Categorical(torch.randn(32, 12))
        ... }) # No error

        >>> spec_dict.validate({
        ...     "obs": {
        ...         "arm":      torch.randn(32, 32), # Wrong shape
        ...         "gripper":  torch.randn(32, 12)
        ...     },
        ...     "action": torch.randn(32, 12),
        ...     "action_dist": torch.distributions.Categorical(torch.randn(32, 12))
        ... }) # raises ValueError

        Filtering input data:
        ---------------------
        >>> input_data = {
        ...     "obs": {
        ...         "arm":      torch.randn(32, 64),
        ...         "gripper":  torch.randn(32, 12),
        ...         "unused":   torch.randn(32, 12)
        ...     },
        ...     "action": torch.randn(32, 12),
        ...     "action_dist": torch.distributions.Categorical(torch.randn(32, 12)),
        ...     "unused": torch.randn(32, 12)
        ... }
        >>> input_data.filter(spec_dict) # returns a dict with only the keys in the spec
        {
            "obs": {
                "arm":      input_data["obs"]["arm"],
                "gripper":  input_data["obs"]["gripper"]
            },
            "action": input_data["action"],
            "action_dist": input_data["action_dist"]
        }

    Raises:
        ValueError: If the data doesn't match the spec.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys_set = set(self.keys())

    def validate(
        self,
        data: DATA_TYPE,
        exact_match: bool = False,
    ) -> None:
        """Checks whether the data matches the spec.

        Args:
            data: The data which should match the spec. It can also be a spec.
            exact_match: If true, the data and the spec must be exactly identical.
                Otherwise, the data is validated as long as it contains at least the
                elements of the spec, but can contain more entries.
        Raises:
            ValueError: If the data doesn't match the spec.
        """
        data = NestedDict(data)
        data_keys_set = set(data.keys())
        missing_keys = self._keys_set.difference(data_keys_set)
        if missing_keys:
            raise ValueError(_MISSING_KEYS_FROM_DATA.format(missing_keys))
        if exact_match:
            data_spec_missing_keys = data_keys_set.difference(self._keys_set)
            if data_spec_missing_keys:
                raise ValueError(_MISSING_KEYS_FROM_SPEC.format(data_spec_missing_keys))

        for spec_name, spec in self.items():
            data_to_validate = data[spec_name]
            if isinstance(spec, TensorSpecs):
                try:
                    spec.validate(data_to_validate)
                except ValueError as e:
                    raise ValueError(
                        f"Mismatch found in data element {spec_name}, which is a TensorSpecs: {e}"
                    )
            elif isinstance(spec, Type):
                if not isinstance(data_to_validate, spec):
                    raise ValueError(
                        _TYPE_MISMATCH.format(
                            spec_name, type(data_to_validate).__name__, spec.__name__
                        )
                    )

    @override(NestedDict)
    def __repr__(self) -> str:
        return f"ModelSpecDict({repr(self._data)})"


def check_specs(
    input_spec: str = "",
    output_spec: str = "",
    filter: bool = True,
    cache: bool = False,
    input_exact_match: bool = False,
    output_exact_match: bool = False,
):
    """A general-purpose [stateful decorator]
    (https://realpython.com/primer-on-python-decorators/#stateful-decorators) to
    enforce input/output specs for any instance method that has `input_dict` in input
    args and returns and a dict.


    It has the ability to filter the input dict to only contain the keys in the spec
    and also to cache to make sure the spec check is only called once in the lifetime
    of the instance.


    Args:
        func: The instance method to decorate. It should be a callable that takes
            `self` as the first argument, `input_dict` as the second argument and any
            other keyword argument thereafter. It should return a dict.
        input_spec: `getattr(self, input_spec)` should correspond to the spec that
            `input_dict` should comply with
        output_spec: `getattr(self, output_spec)` should correspond to the spec that
            the returned dict should comply with.
        filter: If True, the input_dict is filtered by its corresponding spec tree
            structure and then passed into the implemented function to make sure user
            is not confounded by unnecessary data.
        cache: If True, only checks the input/output for the first time the instance
            method is called.
        input_exact_match: If True, the input data must match the spec exactly.
            Otherwise, the data is validated as long as it contains at least the
            elements of the spec, but can contain more entries.
        output_exact_match: If True, the output data must match the spec exactly.
            Otherwise, the data is validated as long as it contains at least the
            elements of the spec, but can contain more entries.

    Returns:
        A wrapped instance method. In case of `cache=True`, after the first invokation
        of the decorated method, the intance will have `__checked_specs_cache__`
        attribute that store which method has been invoked at least once. This is a
        special attribute can be used as for the cache. The wrapped class method also
        has a special attribute `__checked_specs__` that marks the method as decorated.
    """

    if not input_spec and not output_spec:
        raise ValueError("At least one of input_spec or output_spec must be provided.")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, input_dict, **kwargs):

            if not isinstance(input_dict, Mapping):
                raise ValueError(
                    f"input_dict must be a Mapping, got {type(input_dict).__name__}"
                )

            if cache and not hasattr(self, "__checked_specs_cache__"):
                self.__checked_specs_cache__ = {}

            def should_validate():
                return not cache or func.__name__ not in self.__checked_specs_cache__

            input_dict_ = NestedDict(input_dict)

            if input_spec:
                input_spec_ = getattr(self, input_spec)()
                if should_validate():
                    try:
                        input_spec_.validate(input_dict_, exact_match=input_exact_match)
                    except ValueError as e:
                        raise ValueError(f"Input spec validation failed on {self.__class__.__name__}.{func.__name__}, {e}.")
                if filter:
                    input_dict_ = input_dict_.filter(input_spec_)

            output_dict_ = func(self, input_dict_, **kwargs)
            if output_spec and should_validate():
                output_spec_ = getattr(self, output_spec)()
                try:
                    output_spec_.validate(output_dict_, exact_match=output_exact_match)
                except ValueError as e:
                    raise ValueError(f"Output spec validation failed on {self.__class__.__name__}.{func.__name__}, {e}.")

            if cache:
                self.__checked_specs_cache__[func.__name__] = True

            return output_dict_

        wrapper.__check_specs__ = True
        return wrapper

    return decorator
