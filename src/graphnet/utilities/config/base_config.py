"""Base config class(es)."""

from abc import abstractmethod
from collections import OrderedDict
import inspect
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel
import ruamel.yaml as yaml

from graphnet.utilities.logging import LoggerMixin


CONFIG_FILES_SUFFIXES = (".yml", ".yaml")


class BaseConfig(BaseModel):
    """Base class for Configs."""

    @classmethod
    def load(cls, path: str) -> "BaseConfig":
        """Load BaseConfig from `path`."""
        assert path.endswith(
            CONFIG_FILES_SUFFIXES
        ), "Please specify YAML config file."
        with open(path, "r") as f:
            yaml_ = yaml.YAML(typ="safe", pure=True)
            config_dict = yaml_.load(f)

        return cls(**config_dict)

    def dump(self, path: Optional[str] = None) -> Optional[str]:
        """Save BaseConfig to `path` as YAML file, or return as string."""
        config_dict = self._as_dict()[self.__class__.__name__]

        yaml_ = yaml.YAML(typ="safe", pure=True)
        if path:
            if not path.endswith(CONFIG_FILES_SUFFIXES):
                path += CONFIG_FILES_SUFFIXES[0]
            with open(path, "w") as f:
                yaml_.dump(config_dict, f)
            return None
        else:
            return yaml_.dump(config_dict)

    def _as_dict(self) -> Dict[str, Dict[str, Any]]:
        """Represent BaseConfig as a dict.

        This builds on `BaseModel.dict()` but can be overwritten
        """
        return {self.__class__.__name__: self.dict()}


def get_all_argument_values(
    fn: Callable, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    """Return dict of all argument values to `fn`, including defaults."""
    # Get all default argument values
    cfg = OrderedDict()
    for key, parameter in inspect.signature(fn).parameters.items():
        if key == "self" or parameter.default == inspect._empty:
            continue
        cfg[key] = parameter.default

    # Add positional arguments
    for key, val in zip(cfg.keys(), args):
        cfg[key] = val

    # Add keyword arguments
    cfg.update(kwargs)

    return cfg
