import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_MAX_RUN_LOGS = 5


def _sorted_by_mtime_desc(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)


def _select_keep_run_names(
    log_dir: Optional[Path],
    episode_steps_dir: Optional[Path],
    keep: int,
    current_run_log_name: Optional[str] = None,
) -> list[str]:
    keep_names: list[str] = []

    if log_dir is not None and log_dir.exists():
        run_logs = [path for path in log_dir.iterdir() if path.is_file() and path.suffix == ".log" and path.name != "latest.log"]
        keep_names.extend(path.name for path in _sorted_by_mtime_desc(run_logs))
    elif episode_steps_dir is not None and episode_steps_dir.exists():
        run_dirs = [path for path in episode_steps_dir.iterdir() if path.is_dir()]
        keep_names.extend(path.name for path in _sorted_by_mtime_desc(run_dirs))

    if current_run_log_name:
        keep_names = [current_run_log_name] + [name for name in keep_names if name != current_run_log_name]

    unique_names: list[str] = []
    seen: set[str] = set()
    for name in keep_names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
        if len(unique_names) >= keep:
            break
    return unique_names


def prune_r2e_gym_logs(
    log_dir: Optional[Path | str],
    episode_steps_dir: Optional[Path | str],
    keep: int = DEFAULT_MAX_RUN_LOGS,
    current_run_log_name: Optional[str] = None,
) -> list[str]:
    if keep < 1:
        raise ValueError("keep must be at least 1")

    resolved_log_dir = Path(log_dir).expanduser() if log_dir else None
    resolved_episode_steps_dir = Path(episode_steps_dir).expanduser() if episode_steps_dir else None

    keep_run_names = _select_keep_run_names(
        log_dir=resolved_log_dir,
        episode_steps_dir=resolved_episode_steps_dir,
        keep=keep,
        current_run_log_name=current_run_log_name,
    )
    keep_run_name_set = set(keep_run_names)

    if resolved_log_dir is not None and resolved_log_dir.exists():
        for path in resolved_log_dir.iterdir():
            if path.is_dir():
                continue
            if path.suffix == ".log" and path.name != "latest.log" and path.name not in keep_run_name_set:
                path.unlink()

        keep_pid_names = {Path(name).with_suffix(".pid").name for name in keep_run_names}
        for path in resolved_log_dir.iterdir():
            if path.is_dir():
                continue
            if path.suffix == ".pid" and path.name != "latest.pid" and path.name not in keep_pid_names:
                path.unlink()

    if resolved_episode_steps_dir is not None and resolved_episode_steps_dir.exists():
        for path in resolved_episode_steps_dir.iterdir():
            if path.is_dir() and path.name not in keep_run_name_set:
                shutil.rmtree(path)

    return keep_run_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune R2E Gym log directories to the newest N runs.")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--episode-steps-dir", default=None)
    parser.add_argument("--current-run-log-name", default=None)
    parser.add_argument("--keep", type=int, default=DEFAULT_MAX_RUN_LOGS)
    args = parser.parse_args()

    prune_r2e_gym_logs(
        log_dir=args.log_dir,
        episode_steps_dir=args.episode_steps_dir,
        current_run_log_name=args.current_run_log_name,
        keep=args.keep,
    )


if __name__ == "__main__":
    main()