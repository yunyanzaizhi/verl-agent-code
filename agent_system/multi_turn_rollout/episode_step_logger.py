import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from agent_system.multi_turn_rollout.log_retention import DEFAULT_MAX_RUN_LOGS, prune_r2e_gym_logs


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


def _append_block(lines: list[str], title: str, body: Any) -> None:
    lines.append("")
    lines.append("-" * 80)
    lines.append(title)
    lines.append("-" * 80)
    if body is None:
        lines.append("<none>")
    elif isinstance(body, str):
        lines.append(body.rstrip() if body else "<empty>")
    else:
        _append_mapping(lines, body)


def _append_mapping(lines: list[str], value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    value = _json_safe(value)
    if isinstance(value, dict):
        if not value:
            lines.append(prefix + "{}")
            return
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                _append_mapping(lines, item, indent + 2)
            elif isinstance(item, list):
                if not item:
                    lines.append(f"{prefix}{key}: []")
                elif all(not isinstance(entry, (dict, list)) for entry in item) and len(str(item)) <= 160:
                    lines.append(f"{prefix}{key}: {item}")
                else:
                    lines.append(f"{prefix}{key}:")
                    _append_mapping(lines, item, indent + 2)
            elif isinstance(item, str) and "\n" in item:
                lines.append(f"{prefix}{key}:")
                for line in item.rstrip().splitlines():
                    lines.append(f"{prefix}  {line}")
            else:
                lines.append(f"{prefix}{key}: {item}")
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}- [{idx}]")
                _append_mapping(lines, item, indent + 2)
            else:
                lines.append(f"{prefix}- {item}")
        return
    lines.append(prefix + str(value))


def _filtered_env_info(info: Any) -> Dict[str, Any]:
    if not isinstance(info, dict):
        return {}
    hidden = {"raw_model_output", "r2e_action", "r2e_raw_observation"}
    return {key: value for key, value in info.items() if key not in hidden}


def _environment_label(record: Dict[str, Any]) -> str:
    env = record.get("env", {}) or {}
    env_name = str(env.get("name") or "").strip()
    if env_name:
        return env_name.replace("-", "_").upper()

    actor = record.get("actor", {}) or {}
    parsed_action = actor.get("parsed_action") if isinstance(actor, dict) else None
    if isinstance(parsed_action, dict) and "tool_name" in parsed_action:
        return "CODE_REPAIR"
    return "R2E"


def render_step_log(record: Dict[str, Any]) -> str:
    task = record.get("task", {}) or {}
    model_output = record.get("model_output", {}) or {}
    actor = record.get("actor", {}) or {}
    env = record.get("env", {}) or {}

    header = f"{_environment_label(record)} EPISODE STEP"
    lines = [
        "=" * 80,
        header,
        "=" * 80,
        f"train_step: {record.get('train_step')}",
        f"episode: {record.get('episode')}",
        f"step: {record.get('step')}",
    ]
    _append_block(lines, "TASK", task)
    _append_block(lines, "MODEL OUTPUT (RAW)", model_output.get("raw_response_text", ""))
    _append_block(
        lines,
        "ACTOR",
        {
            "is_action_valid": actor.get("is_action_valid"),
            "parsed_action": actor.get("parsed_action"),
        },
    )
    _append_block(
        lines,
        "ENVIRONMENT",
        {
            "reward": env.get("reward"),
            "done": env.get("done"),
            "raw_observation": env.get("raw_observation"),
            "next_observation_for_model": env.get("observation"),
            "anchor": env.get("anchor"),
            "info": _filtered_env_info(env.get("info", {})),
        },
    )
    _append_block(lines, "PROTOCOL", env.get("protocol"))
    lines.append("")
    return "\n".join(lines)


class EpisodeStepLogger:
    def __init__(self, root_dir: Optional[Any], run_log_name: Optional[str], enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.root_dir = Path(root_dir).expanduser() if root_dir else None
        self.run_log_name = _safe_filename_component(run_log_name)

        if self.enabled and self.root_dir is not None and self.root_dir.parent.name == "episode_steps":
            prune_r2e_gym_logs(
                log_dir=self.root_dir.parent.parent,
                episode_steps_dir=self.root_dir.parent,
                current_run_log_name=self.run_log_name,
                keep=DEFAULT_MAX_RUN_LOGS,
            )

    @classmethod
    def from_config(cls, config: Any) -> "EpisodeStepLogger":
        env_name = str(_cfg_get(config, "env.env_name", "") or "").lower()
        if env_name in {"code_repair", "leetcode_repair", "leetcode_code_repair"} or "code_repair" in env_name:
            prefix = "env.code_repair"
            enabled_env = "CODE_REPAIR_STEP_LOG_ENABLED"
            root_env = "CODE_REPAIR_STEP_LOG_DIR"
            name_env = "CODE_REPAIR_RUN_LOG_NAME"
            default_log_root = "code_repair"
        else:
            prefix = "env.r2e_gym"
            enabled_env = "R2E_STEP_LOG_ENABLED"
            root_env = "R2E_STEP_LOG_DIR"
            name_env = "R2E_RUN_LOG_NAME"
            default_log_root = "r2e_gym"
        enabled = _to_bool(
            _cfg_get(config, f"{prefix}.step_log_enabled", os.environ.get(enabled_env)),
            default=False,
        )
        root_dir = _cfg_get(config, f"{prefix}.step_log_dir", os.environ.get(root_env))
        run_log_name = _cfg_get(config, f"{prefix}.run_log_name", os.environ.get(name_env, "run.log"))
        if enabled and not root_dir:
            root_dir = Path("logs") / default_log_root / "episode_steps" / _safe_filename_component(run_log_name)
        return cls(root_dir=root_dir, run_log_name=run_log_name, enabled=enabled)

    def step_directory(self, train_step: int, episode: int) -> Path:
        return Path(f"train_step_{int(train_step):06d}") / f"episode_{int(episode):06d}"

    def step_filename(self, train_step: int, episode: int, step: int) -> str:
        return f"step_{int(step):06d}.log"

    def step_path(self, train_step: int, episode: int, step: int) -> Path:
        return self.step_directory(train_step=train_step, episode=episode) / self.step_filename(
            train_step=train_step,
            episode=episode,
            step=step,
        )

    def write_step(self, train_step: int, episode: int, step: int, payload: Dict[str, Any]) -> Optional[Path]:
        if not self.enabled or self.root_dir is None:
            return None

        path = self.root_dir / self.step_path(train_step=train_step, episode=episode, step=step)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "train_step": int(train_step),
            "episode": int(episode),
            "step": int(step),
        }
        record.update(payload)
        record.pop("model_input", None)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(render_step_log(_json_safe(record)), encoding="utf-8")
        tmp_path.replace(path)
        return path
