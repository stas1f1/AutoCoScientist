"""Class injection mechanism inspired by Hydra's config system.

This module provides utilities for dynamic class loading, instantiation, and configuration
management similar to Hydra's object instantiation capabilities.
"""

import importlib
import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

from pydantic import BaseModel, ConfigDict, PrivateAttr

T = TypeVar("T")


class ClassConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    """Configuration for class instantiation.

    Similar to Hydra's _target_ and _partial_ fields, this provides a flexible
    way to specify which class to instantiate and how to configure it.
    """

    _target_: str  # Full import path to the class (e.g., "module.submodule.ClassName")
    _partial_: bool = False  # Whether to return a partial function instead of instance
    _recursive_: bool = True  # Whether to recursively instantiate nested configs


class ClassRegistry(BaseModel):
    """Registry for managing class configurations and instances."""

    _instances: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _configs: Dict[str, ClassConfig] = PrivateAttr(default_factory=dict)
    _aliases: Dict[str, str] = PrivateAttr(default_factory=dict)

    def register_class(self, alias: str, class_path: str, **config_kwargs) -> None:
        """Register a class with an alias and default configuration.

        Args:
            alias: Short alias for the class
            class_path: Full import path to the class
            **config_kwargs: Default configuration parameters
        """
        self._aliases[alias] = class_path
        self._configs[alias] = ClassConfig(_target_=class_path, **config_kwargs)

    def get_class_path(self, identifier: str) -> str:
        """Get the full class path from an alias or return the identifier if it's already a path.

        Args:
            identifier: Either an alias or a full class path

        Returns:
            Full class path to the class
        """
        return self._aliases.get(identifier, identifier)

    def get_config(self, identifier: str) -> Optional[ClassConfig]:
        """Get configuration for a class by alias.

        Args:
            identifier: Class alias

        Returns:
            ClassConfig if found, None otherwise
        """
        return self._configs.get(identifier)


class ClassInjector:
    """Main class for dynamic class loading and instantiation."""

    def __init__(self, registry: Optional[ClassRegistry] = None):
        """Initialize the class injector.

        Args:
            registry: Optional class registry for managing aliases and configs
        """
        self.registry = registry or ClassRegistry()
        self._cache: Dict[str, Type] = {}

    def load_class(self, class_path: str) -> Type:
        """Dynamically load a class from its import path.

        Args:
            class_path: Full import path to the class (e.g., "module.submodule.ClassName")

        Returns:
            The loaded class

        Raises:
            ImportError: If the module or class cannot be imported
            AttributeError: If the class doesn't exist in the module
        """
        if class_path in self._cache:
            return self._cache[class_path]

        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            self._cache[class_path] = cls
            return cls
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load class '{class_path}': {e}")

    def instantiate(
        self, config: Union[ClassConfig, Dict[str, Any], str], **override_kwargs
    ) -> Any:
        """Instantiate a class from configuration.

        Args:
            config: Class configuration (ClassConfig, dict, or alias string)
            **override_kwargs: Parameters to override in the configuration

        Returns:
            Instantiated object or partial function if _partial_ is True

        Raises:
            TypeError: If the configuration is invalid
            ImportError: If the class cannot be loaded
        """
        # Convert config to ClassConfig if needed
        if isinstance(config, str):
            # Check if it's an alias
            class_path = self.registry.get_class_path(config)
            config = ClassConfig(_target_=class_path)
        elif isinstance(config, dict):
            config = ClassConfig(**config)
        elif not isinstance(config, ClassConfig):
            raise TypeError(f"Invalid config type: {type(config)}")

        # Load the class
        config_dict = config.model_dump()
        cls = self.load_class(config_dict["_target_"])

        # Prepare instantiation parameters
        params = self._prepare_params(config, override_kwargs)

        # Handle partial instantiation
        if config_dict.get("_partial_", False):
            return lambda **kwargs: cls(**{**params, **kwargs})

        # Instantiate the class
        try:
            return cls(**params)
        except TypeError as e:
            # Try to provide more helpful error messages
            sig = inspect.signature(cls.__init__)
            required_params = [
                name
                for name, param in sig.parameters.items()
                if param.default == inspect.Parameter.empty and name != "self"
            ]
            raise TypeError(
                f"Failed to instantiate {config_dict['_target_']}: {e}. "
                f"Required parameters: {required_params}. "
                f"Provided parameters: {list(params.keys())}"
            )

    def _prepare_params(
        self, config: ClassConfig, override_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for class instantiation.

        Args:
            config: Class configuration
            override_kwargs: Parameters to override

        Returns:
            Dictionary of parameters ready for instantiation
        """
        params: Dict[str, Any] = {}

        # Get all non-private attributes from config
        config_dict = config.model_dump()
        for key, value in config_dict.items():
            if not key.startswith("_") and key != "url_pattern":
                params[key] = value

        # Apply overrides
        params.update(override_kwargs)

        # Recursively instantiate nested configs if enabled
        if config_dict.get("_recursive_", True):
            params = self._recursive_instantiate(params)

        return params

    def _recursive_instantiate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively instantiate nested configurations.

        Args:
            params: Dictionary of parameters that may contain nested configs

        Returns:
            Dictionary with nested configs instantiated
        """
        result = {}

        for key, value in params.items():
            if isinstance(value, dict) and "_target_" in value:
                # This is a nested config, instantiate it
                result[key] = self.instantiate(value)
            elif isinstance(value, list):
                # Handle lists that may contain configs
                result[key] = [
                    self.instantiate(item)
                    if isinstance(item, dict) and "_target_" in item
                    else item
                    for item in value
                ]
            else:
                result[key] = value

        return result

    def get_signature(self, class_path: str) -> inspect.Signature:
        """Get the signature of a class constructor.

        Args:
            class_path: Full import path to the class

        Returns:
            Signature of the class constructor
        """
        cls = self.load_class(class_path)
        return inspect.signature(cls.__init__)

    def get_type_hints(self, class_path: str) -> Dict[str, Any]:
        """Get type hints for a class constructor.

        Args:
            class_path: Full import path to the class

        Returns:
            Dictionary of parameter names to their type hints
        """
        cls = self.load_class(class_path)
        return get_type_hints(cls.__init__)


# Global registry instance
_default_registry = ClassRegistry()


# Convenience functions
def register_class(alias: str, class_path: str, **config_kwargs) -> None:
    """Register a class with the default registry.

    Args:
        alias: Short alias for the class
        class_path: Full import path to the class
        **config_kwargs: Default configuration parameters
    """
    _default_registry.register_class(alias, class_path, **config_kwargs)


def instantiate(
    config: Union[ClassConfig, Dict[str, Any], str], **override_kwargs
) -> Any:
    """Instantiate a class using the default injector.

    Args:
        config: Class configuration (ClassConfig, dict, or alias string)
        **override_kwargs: Parameters to override in the configuration

    Returns:
        Instantiated object or partial function if _partial_ is True
    """
    injector = ClassInjector(_default_registry)
    return injector.instantiate(config, **override_kwargs)


def load_class(class_path: str) -> Type:
    """Load a class using the default injector.

    Args:
        class_path: Full import path to the class

    Returns:
        The loaded class
    """
    injector = ClassInjector(_default_registry)
    return injector.load_class(class_path)


# Example usage and configuration helpers
def create_adapter_config(
    class_path: str, url_pattern: str, **config_kwargs
) -> Dict[str, Any]:
    """Create a configuration dictionary for repository adapters.

    Args:
        class_path: Full import path to the adapter class
        url_pattern: Regex pattern to match URLs for this adapter
        **config_kwargs: Additional configuration parameters

    Returns:
        Configuration dictionary ready for use with ClassInjector
    """
    return {"_target_": class_path, "url_pattern": url_pattern, **config_kwargs}


def create_service_config(
    adapters: List[Dict[str, Any]], **service_kwargs
) -> Dict[str, Any]:
    """Create a configuration dictionary for services with multiple adapters.

    Args:
        adapters: List of adapter configurations
        **service_kwargs: Additional service configuration parameters

    Returns:
        Configuration dictionary for service instantiation
    """
    return {"adapters": adapters, **service_kwargs}
