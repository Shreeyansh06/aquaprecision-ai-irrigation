"""
AquaPrecision grader functions — one per task.

Rules enforced per cloud validator requirements:
  - All signatures have default None args (survives parameterless reflection)
  - All return values are pure Python float strictly between 0 and 1
  - No numpy types, no boundary values (0.0 or 1.0)
"""


def _clamp(value) -> float:
    """Return a pure Python float strictly between 0.02 and 0.95."""
    try:
        v = float(value)
        if v != v:  # NaN
            v = 0.5
    except (TypeError, ValueError):
        v = 0.5
    return float(max(0.02, min(0.95, round(v, 4))))


def _score_trajectory(trajectory: dict) -> float:
    """
    Common grading logic shared by all tasks.
    Formula: (survival_rate * 0.6) + (avg_health * 0.3) + (water_efficiency * 0.1)
    """
    trajectory = trajectory or {}

    # Try final_observation first, then observation, then last step
    final_obs = (
        trajectory.get("final_observation")
        or trajectory.get("observation")
        or {}
    )

    # Also try extracting from steps list
    steps = trajectory.get("steps", [])
    if not final_obs and steps:
        final_obs = steps[-1].get("observation", {})

    field = final_obs.get("field", [])
    if not field:
        return 0.1  # fallback — not 0.0

    total = len(field)
    alive = [c for c in field if not c.get("is_dead", True)]
    survival_rate = len(alive) / total if total > 0 else 0.0

    avg_health = (
        sum(c.get("crop_health", 0.0) for c in field) / total
        if total > 0 else 0.0
    )

    water = final_obs.get("water_tank", {})
    current = float(water.get("current", 0))
    capacity = float(water.get("capacity", 1))
    water_efficiency = current / capacity if capacity > 0 else 0.0

    raw = (survival_rate * 0.6) + (avg_health * 0.3) + (water_efficiency * 0.1)
    return _clamp(raw)


def easy_grader(trajectory: dict = None) -> float:
    """Grade the easy task (task_easy_1): single cell, 10 steps."""
    return _score_trajectory(trajectory or {})


def medium_grader(trajectory: dict = None) -> float:
    """Grade the medium task (task_medium_1): 3x3 grid, 20 steps."""
    return _score_trajectory(trajectory or {})


def hard_grader(trajectory: dict = None) -> float:
    """Grade the hard task (task_hard_1): 5x5 grid, 50 steps."""
    return _score_trajectory(trajectory or {})
