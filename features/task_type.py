"""Task category one-hot features."""
from data.taxonomy import TaskType, classify_task


def task_type_features(prompt: str) -> dict[str, float]:
    selected = classify_task(prompt)
    return {f"task_{task.value}": float(task == selected) for task in TaskType}
