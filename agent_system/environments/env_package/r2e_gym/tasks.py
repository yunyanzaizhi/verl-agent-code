import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class R2EGymTask:
    task_id: str
    dataset_name: str
    split: str
    index: int
    repo_name: str
    docker_image: str
    commit_hash: str
    problem_statement: str
    expected_output_json: str
    relevant_files: List[str]
    raw_record: Dict[str, Any]


def _to_plain_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "items"):
        return dict(record.items())
    raise TypeError(f"Unsupported R2E task record type: {type(record)!r}")


def _extract_issue(problem_statement: str) -> str:
    text = problem_statement or ""
    match = re.search(r"\[ISSUE\](.*?)\[/ISSUE\]", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip()


def normalize_r2e_task_record(
    record: Dict[str, Any],
    dataset_name: str,
    split: str,
    index: int,
) -> R2EGymTask:
    row = _to_plain_dict(record)
    repo_name = str(row.get("repo_name") or row.get("repo") or "unknown_repo")
    docker_image = str(row.get("docker_image") or row.get("image_name") or "")
    commit_hash = str(row.get("commit_hash") or row.get("base_commit") or row.get("instance_id") or index)
    problem_statement = _extract_issue(str(row.get("problem_statement") or row.get("prompt") or ""))
    expected_output_json = str(row.get("expected_output_json") or "{}")
    relevant_files = list(row.get("relevant_files") or [])
    task_id = f"{dataset_name}:{split}:{index}:{commit_hash}"
    return R2EGymTask(
        task_id=task_id,
        dataset_name=dataset_name,
        split=split,
        index=index,
        repo_name=repo_name,
        docker_image=docker_image,
        commit_hash=commit_hash,
        problem_statement=problem_statement,
        expected_output_json=expected_output_json,
        relevant_files=relevant_files,
        raw_record=row,
    )


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def load_r2e_tasks_from_config(config: Any, is_train: bool = True) -> List[R2EGymTask]:
    dataset_key = "train_dataset" if is_train else "val_dataset"
    split_key = "train_split" if is_train else "val_split"
    max_key = "max_train_tasks" if is_train else "max_val_tasks"
    dataset_name = _cfg_get(config, dataset_key) or _cfg_get(config, "dataset")
    split = _cfg_get(config, split_key) or ("train" if is_train else "validation")
    if not dataset_name:
        raise ValueError("env.r2e_gym train_dataset/val_dataset must be configured.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to load R2E-Gym tasks.") from exc

    dataset = load_dataset(str(dataset_name), split=str(split))
    max_tasks: Optional[int] = _cfg_get(config, max_key)
    if max_tasks is not None:
        max_tasks = int(max_tasks)
    tasks = []
    for idx, row in enumerate(dataset):
        if max_tasks is not None and idx >= max_tasks:
            break
        tasks.append(normalize_r2e_task_record(row, str(dataset_name), str(split), idx))
    if not tasks:
        raise ValueError(f"No R2E-Gym tasks loaded from {dataset_name}:{split}.")
    return tasks
