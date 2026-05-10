import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._=-]+")


def _cfg_get(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for key in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_filename_component(value: Any) -> str:
    text = os.path.basename(str(value or "run.log"))
    text = _UNSAFE_FILENAME_CHARS.sub("_", text).strip("._")
    return text or "run.log"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return _json_safe(value.tolist())
        if isinstance(value, np.generic):
            return _json_safe(value.item())
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _json_safe(value.detach().cpu().tolist())
    except Exception:
        pass
    return str(value)


class EpisodeStepLogger:
    def __init__(self, root_dir: Optional[Any], run_log_name: Optional[str], enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.root_dir = Path(root_dir).expanduser() if root_dir else None
        self.run_log_name = _safe_filename_component(run_log_name)

    @classmethod
    def from_config(cls, config: Any) -> "EpisodeStepLogger":
        enabled = _to_bool(
            _cfg_get(config, "env.r2e_gym.step_log_enabled", os.environ.get("R2E_STEP_LOG_ENABLED")),
            default=False,
        )
        root_dir = _cfg_get(config, "env.r2e_gym.step_log_dir", os.environ.get("R2E_STEP_LOG_DIR"))
        run_log_name = _cfg_get(config, "env.r2e_gym.run_log_name", os.environ.get("R2E_RUN_LOG_NAME", "run.log"))
        if enabled and not root_dir:
            root_dir = Path("logs") / "r2e_gym" / "episode_steps" / _safe_filename_component(run_log_name)
        return cls(root_dir=root_dir, run_log_name=run_log_name, enabled=enabled)

    def step_filename(self, train_step: int, episode: int, step: int) -> str:
        return (
            f"{self.run_log_name}"
            f"-train_step_{int(train_step):06d}"
            f"-episode_{int(episode):06d}"
            f"-step_{int(step):06d}.json"
        )

    def write_step(self, train_step: int, episode: int, step: int, payload: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled or self.root_dir is None:
            return None

        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.root_dir / self.step_filename(train_step=train_step, episode=episode, step=step)
        record = {
            "train_step": int(train_step),
            "episode": int(episode),
            "step": int(step),
        }
        record.update(payload)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(_json_safe(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        tmp_path.replace(path)
        return path
