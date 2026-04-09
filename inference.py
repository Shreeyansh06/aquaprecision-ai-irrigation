"""
AquaPrecision: AI Irrigation Simulator — OpenEnv Inference Script

Structured stdout logging follows the required [START] / [STEP] / [END] format.

Required environment variables:
  APP_URL      — Base URL of the deployed HF Space (e.g. https://<space>.hf.space)
  API_BASE_URL — LLM API base URL (e.g. https://generativelanguage.googleapis.com/v1beta/openai/)
  MODEL_NAME   — LLM model identifier (e.g. gemini-2.0-flash)
  HF_TOKEN     — API token for authenticated inference
  OPENAI_API_KEY — OpenAI/Gemini key
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_URL = os.getenv("APP_URL", os.getenv("SPACE_URL", "http://localhost:7860")).rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Build OpenAI-compatible client
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
elif API_BASE_URL and OPENAI_API_KEY:
    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)
elif OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
else:
    client = None
    print(
        "WARNING: No LLM credentials found. Falling back to heuristic agent.",
        file=sys.stderr,
    )

SEED = "openenv_inference_seed_42"


# ---------------------------------------------------------------------------
# ✅ Score clamping — strictly between 0 and 1
# ---------------------------------------------------------------------------

def clamp_score(value: float) -> float:
    """Ensure score is strictly between 0.0 and 1.0 (exclusive)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0.5
    return round(min(0.99, max(0.01, v)), 4)


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log_start(task: dict, seed: str) -> None:
    payload = {
        "task_id": task["id"],
        "task_name": task.get("name", task["id"]),
        "difficulty": task.get("difficulty", "unknown"),
        "max_steps": task.get("max_steps", 24),
        "seed": seed,
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, obs: dict) -> None:
    reward = clamp_score(reward)  # ✅ always clamped
    field = obs.get("field", [])
    avg_health = (
        round(sum(c.get("crop_health", 0.5) for c in field) / len(field), 4)
        if field else 0.5
    )
    water = obs.get("water_tank", {})
    payload = {
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "water_remaining": round(water.get("current", 0), 2),
        "avg_health": clamp_score(avg_health),  # ✅ clamped
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(task_id: str, score: float, total_steps: int, total_reward: float) -> None:
    score = clamp_score(score)  # ✅ always clamped
    payload = {
        "task_id": task_id,
        "score": score,
        "total_steps": total_steps,
        "total_reward": round(total_reward, 4),
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Heuristic fallback agent
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict) -> dict:
    field = obs.get("field", [])
    alive = [c for c in field if not c.get("is_dead", False)]
    if not alive:
        return {"type": "wait"}
    driest = min(alive, key=lambda c: c.get("moisture", 1))
    water = obs.get("water_tank", {})
    if driest.get("moisture", 1) < 0.4 and water.get("current", 0) >= 5:
        return {"type": "irrigate", "cell_id": driest["id"], "amount": 5}
    return {"type": "wait"}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def llm_action(obs: dict, task_description: str) -> dict:
    if client is None:
        return heuristic_action(obs)

    field = obs.get("field", [])
    weather = obs.get("weather", {})
    water = obs.get("water_tank", {})

    prompt = (
        f"Task: {task_description}\n"
        f"Step: {obs.get('step', 0)}\n"
        f"Weather: temperature={weather.get('temperature', 25):.1f}C, "
        f"humidity={weather.get('humidity', 50):.1f}%, "
        f"raining={weather.get('is_raining', False)}, "
        f"evaporation={weather.get('evaporation_rate', 0.01):.4f}\n"
        f"Water tank: {water.get('current', 0):.1f} / {water.get('capacity', 1000):.1f} L\n"
        f"Field cells:\n"
        + "\n".join(
            f"  cell {c['id']}: moisture={c.get('moisture', 0):.2f}, "
            f"health={c.get('crop_health', 0):.2f}, dead={c.get('is_dead', False)}"
            for c in field
        )
        + "\n\n"
        "Goal: Maximise crop health and survival while conserving water.\n"
        "Respond with ONLY a JSON object — no explanation:\n"
        '  {"type": "irrigate", "cell_id": <int>, "amount": <float 0-10>}\n'
        '  or {"type": "wait"}'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI irrigation agent managing a drought-prone farm. "
                        "Return only valid JSON with no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=64,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        action = json.loads(raw)
        if action.get("type") not in ("irrigate", "wait"):
            raise ValueError(f"Unexpected action type: {action}")
        return action
    except Exception as exc:
        print(f"LLM error (falling back to heuristic): {exc}", file=sys.stderr)
        return heuristic_action(obs)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict) -> float:
    log_start(task, SEED)

    try:
        reset_resp = requests.post(
            f"{APP_URL}/api/reset",
            json={"task_id": task["id"], "seed": SEED},
            timeout=30,
        )
        reset_resp.raise_for_status()
        data = reset_resp.json()
    except Exception as exc:
        print(f"ERROR resetting task {task['id']}: {exc}", file=sys.stderr)
        log_end(task["id"], 0.01, 0, 0.0)
        return 0.01

    session_id = data["session_id"]
    obs = data["observation"]

    done = False
    total_reward = 0.0
    steps_taken = 0

    while not done:
        action = llm_action(obs, task.get("description", task["id"]))

        try:
            step_resp = requests.post(
                f"{APP_URL}/api/step",
                json={"session_id": session_id, "action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            result = step_resp.json()
        except Exception as exc:
            print(f"ERROR stepping task {task['id']}: {exc}", file=sys.stderr)
            break

        obs = result["observation"]
        # ✅ always clamp reward
        reward = clamp_score(result.get("reward", 0.1))
        done = result["done"]
        total_reward += reward
        steps_taken += 1

        log_step(obs["step"], action, reward, done, obs)

    # Grade
    try:
        grade_resp = requests.get(
            f"{APP_URL}/api/grade/{session_id}",
            timeout=30,
        )
        grade_resp.raise_for_status()
        raw_score = grade_resp.json().get("score", 0.5)
        # ✅ always clamp final score
        score = clamp_score(raw_score)
    except Exception as exc:
        print(f"ERROR grading task {task['id']}: {exc}", file=sys.stderr)
        score = clamp_score(total_reward / max(steps_taken, 1))

    log_end(task["id"], score, steps_taken, total_reward)
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        tasks_resp = requests.get(f"{APP_URL}/api/tasks", timeout=30)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()
    except Exception as exc:
        print(f"ERROR: Could not fetch tasks from {APP_URL}/api/tasks — {exc}", file=sys.stderr)
        print("Make sure APP_URL or SPACE_URL is set correctly.", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task in tasks:
        try:
            score = run_task(task)
        except Exception as exc:
            print(f"ERROR running task {task['id']}: {exc}", file=sys.stderr)
            score = 0.01
        scores[task["id"]] = score

    print("\n--- SUMMARY ---", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}", flush=True)
