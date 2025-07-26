"""
Central configuration loader/validator for ADAPTIVE_LOSS_PROJECT.
Merges defaults, overrides from experiments/configs/*.yaml, and CLI args.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    def __init__(self,
                 project_root: str = ".",
                 config_dir: str = "experiments/configs"):
        self.project_root = Path(project_root).resolve()
        self.config_dir = self.project_root / config_dir
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, Dict[str, Any]]:
        templates = {}
        for file in self.config_dir.glob("*.yaml"):
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
                # assume list of configs; index by `name`
                for cfg in data:
                    templates[cfg['name']] = cfg
        return templates

    def get(self, name: str) -> Dict[str, Any]:
        if name not in self.templates:
            raise KeyError(f"Config '{name}' not found in {self.config_dir}")
        return self.templates[name]

    def merge(self,
              base: Dict[str, Any],
              overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep‐merge override into base.
        """
        result = dict(base)
        for k, v in overrides.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self.merge(result[k], v)
            else:
                result[k] = v
        return result
