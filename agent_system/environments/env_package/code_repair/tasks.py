import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class CodeRepairTask:
    task_id: str
    dataset_name: str
    split: str
    index: int
    dataset_task_id: str
    question_id: Any
    difficulty: str
    tags: List[str]
    problem_description: str
    support_code: str
    starter_code: str
    current_starter_code: str
    completion: str
    entry_point: str
    test_code: str
    visible_test_code: str
    visible_examples: List[Dict[str, Any]]
    raw_record: Dict[str, Any]


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    if hasattr(config, "get"):
        try:
            return config.get(name, default)
        except Exception:
            pass
    return getattr(config, name, default)


def _to_plain_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "items"):
        return dict(record.items())
    raise TypeError(f"Unsupported code repair task record type: {type(record)!r}")


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def ensure_executable_starter_code(code: str) -> str:
    text = (code or "").rstrip()
    if not text:
        text = "class Solution:\n    pass"
    try:
        ast.parse(text)
        return text + "\n"
    except (IndentationError, SyntaxError) as exc:
        message = str(exc).lower()
        if "expected an indented block" not in message and not text.endswith(":"):
            raise

    lines = text.splitlines()
    if not lines:
        return "class Solution:\n    pass\n"
    last_line = lines[-1]
    indent = len(last_line) - len(last_line.lstrip()) + 4
    lines.append(" " * indent + "pass")
    normalized = "\n".join(lines) + "\n"
    ast.parse(normalized)
    return normalized


def _count_asserts(test_code: str) -> int:
    try:
        module = ast.parse(test_code or "")
    except SyntaxError:
        return 0
    return sum(isinstance(node, ast.Assert) for node in ast.walk(module))


def extract_visible_test_code(test_code: str, visible_test_count: int) -> str:
    if visible_test_count <= 0:
        return test_code or "def check(candidate):\n    pass\n"
    try:
        module = ast.parse(test_code or "")
    except SyntaxError:
        return test_code or "def check(candidate):\n    pass\n"

    new_body = []
    remaining = int(visible_test_count)
    changed = False
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "check":
            body = []
            for statement in node.body:
                if isinstance(statement, ast.Assert):
                    if remaining <= 0:
                        changed = True
                        continue
                    remaining -= 1
                body.append(statement)
            if not body:
                body = [ast.Pass()]
            node = ast.FunctionDef(
                name=node.name,
                args=node.args,
                body=body,
                decorator_list=node.decorator_list,
                returns=node.returns,
                type_comment=getattr(node, "type_comment", None),
            )
        new_body.append(node)
    if not changed and _count_asserts(test_code) <= visible_test_count:
        return test_code or "def check(candidate):\n    pass\n"
    visible_module = ast.Module(body=new_body, type_ignores=[])
    ast.fix_missing_locations(visible_module)
    return ast.unparse(visible_module) + "\n"


def _normalize_examples(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    examples = []
    for item in value:
        if isinstance(item, dict):
            examples.append(dict(item))
    return examples


def normalize_code_repair_record(
    record: Dict[str, Any],
    dataset_name: str,
    split: str,
    index: int,
    visible_test_count: int = 6,
) -> CodeRepairTask:
    row = _to_plain_dict(record)
    dataset_task_id = str(row.get("task_id") or row.get("slug") or row.get("question_id") or index)
    question_id = row.get("question_id", dataset_task_id)
    starter_code = str(row.get("starter_code") or "")
    current_starter_code = ensure_executable_starter_code(starter_code)
    test_code = str(row.get("test") or row.get("test_code") or "def check(candidate):\n    pass\n")
    visible_test_code = extract_visible_test_code(test_code, int(visible_test_count))
    task_id = f"{dataset_name}:{split}:{index}:{dataset_task_id}"
    return CodeRepairTask(
        task_id=task_id,
        dataset_name=str(dataset_name),
        split=str(split),
        index=int(index),
        dataset_task_id=dataset_task_id,
        question_id=question_id,
        difficulty=str(row.get("difficulty") or ""),
        tags=_normalize_tags(row.get("tags")),
        problem_description=str(row.get("problem_description") or row.get("query") or ""),
        support_code=str(row.get("prompt") or ""),
        starter_code=starter_code,
        current_starter_code=current_starter_code,
        completion=str(row.get("completion") or ""),
        entry_point=str(row.get("entry_point") or "Solution()"),
        test_code=test_code,
        visible_test_code=visible_test_code,
        visible_examples=_normalize_examples(row.get("input_output")),
        raw_record=row,
    )


def load_code_repair_tasks_from_jsonl(
    path: str | Path,
    dataset_name: str,
    split: str,
    max_tasks: Optional[int] = None,
    visible_test_count: int = 6,
) -> List[CodeRepairTask]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"CodeRepair dataset path does not exist: {source}")
    tasks: List[CodeRepairTask] = []
    with source.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_tasks is not None and len(tasks) >= int(max_tasks):
                break
            if not line.strip():
                continue
            record = json.loads(line)
            tasks.append(
                normalize_code_repair_record(
                    record,
                    dataset_name=dataset_name,
                    split=split,
                    index=idx,
                    visible_test_count=visible_test_count,
                )
            )
    if not tasks:
        raise ValueError(f"No CodeRepair tasks loaded from {source}.")
    return tasks


def load_code_repair_tasks_from_config(config: Any, is_train: bool = True) -> List[CodeRepairTask]:
    path_key = "train_path" if is_train else "val_path"
    split_key = "train_split" if is_train else "val_split"
    max_key = "max_train_tasks" if is_train else "max_val_tasks"
    path = _cfg_get(config, path_key) or _cfg_get(config, "path")
    if not path:
        raise ValueError(f"env.code_repair.{path_key} must be configured.")
    split = str(_cfg_get(config, split_key, "train" if is_train else "test"))
    dataset_name = str(_cfg_get(config, "dataset_name", "LeetCodeDataset"))
    max_tasks = _cfg_get(config, max_key, None)
    visible_test_count = int(_cfg_get(config, "visible_test_count", 6))
    return load_code_repair_tasks_from_jsonl(
        path=path,
        dataset_name=dataset_name,
        split=split,
        max_tasks=None if max_tasks is None else int(max_tasks),
        visible_test_count=visible_test_count,
    )


def count_test_assertions(test_code: str) -> int:
    return _count_asserts(test_code)
