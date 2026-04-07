"""
AquaPrecision: AI Irrigation Simulator — OpenEnv Inference Script

Structured stdout logging follows the required [START] / [STEP] / [END] format.

Required environment variables:
  APP_URL      — Base URL of the deployed HF Space (e.g. https://<space>.hf.space)
  API_BASE_URL — LLM API base URL (e.g. https://api-inference.huggingface.co/v1)
  MODEL_NAME   — LLM model identifier (e.g. meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN     — Hugging Face API token for authenticated inference

Optional:
  OPENAI_API_KEY — OpenAI key (used if API_BASE_URL is not set)
"""

import os
import sys
import json
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_URL = os.getenv("APP_URL", os.getenv("SPACE_URL", "http://localhost:7860"))
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Build OpenAI-compatible client
# Priority: API_BASE_URL (HF / custom endpoint) > OPENAI_API_KEY
if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )
elif OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
    if not API_BASE_URL:
        MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
else:
    client = None
    print(
        "WARNING: No LLM credentials found. "
        "Set API_BASE_URL + HF_TOKEN or OPENAI_API_KEY. "
        "Falling back to heuristic agent.",
        file=sys.stderr,
    )

SEED = "openenv_inference_seed_42"


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log_start(task: dict, seed: str) -> None:
    """Emit the [START] log line."""
    payload = {
        "task_id": task["id"],
        "task_name": task["name"],
        "difficulty": task["difficulty"],
        "max_steps": task["max_steps"],
        "seed": seed,
    }
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, obs: dict) -> None:
    """Emit a [STEP] log line."""
    payload = {
        "step": step,
        "action": action,
        "reward": round(reward, 4),
        "done": done,
        "water_remaining": round(obs["water_tank"]["current"], 2),
        "avg_health": round(
            sum(c["crop_health"] for c in obs["field"]) / len(obs["field"]), 4
        ),
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(task_id: str, score: float, total_steps: int, total_reward: float) -> None:
    """Emit the [END] log line."""
    payload = {
        "task_id": task_id,
        "score": round(score, 4),
        "total_steps": total_steps,
        "total_reward": round(total_reward, 4),
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# Heuristic fallback agent
# ---------------------------------------------------------------------------

def heuristic_action(obs: dict) -> dict:
    """Simple heuristic: irrigate the driest alive cell if moisture < 0.4."""
    alive = [c for c in obs["field"] if not c["is_dead"]]
    if not alive:
        return {"type": "wait"}
    driest = min(alive, key=lambda c: c["moisture"])
    if driest["moisture"] < 0.4 and obs["water_tank"]["current"] >= 5:
        return {"type": "irrigate", "cell_id": driest["id"], "amount": 5}
    return {"type": "wait"}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def llm_action(obs: dict, task_description: str) -> dict:
    """Ask the LLM for the next action given the current observation."""
    if client is None:
        return heuristic_action(obs)

    prompt = (
        f"Task: {task_description}\n"
        f"Step: {obs['step']}\n"
        f"Weather: temperature={obs['weather']['temperature']:.1f}°C, "
        f"humidity={obs['weather']['humidity']:.1f}%, "
        f"raining={obs['weather']['is_raining']}, "
        f"evaporation={obs['weather']['evaporation_rate']:.4f}\n"
        f"Water tank: {obs['water_tank']['current']:.1f} / {obs['water_tank']['capacity']:.1f} L\n"
        f"Field cells:\n"
        + "\n".join(
            f"  cell {c['id']}: moisture={c['moisture']:.2f}, "
            f"health={c['crop_health']:.2f}, dead={c['is_dead']}"
            for c in obs["field"]
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
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        action = json.loads(raw)
        # Validate basic structure
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

    # Reset environment
    reset_resp = requests.post(
        f"{APP_URL}/api/reset",
        json={"task_id": task["id"], "seed": SEED},
        timeout=30,
    )
    reset_resp.raise_for_status()
    data = reset_resp.json()
    session_id = data["session_id"]
    obs = data["observation"]

    done = False
    total_reward = 0.0
    steps_taken = 0

    while not done:
        action = llm_action(obs, task["description"])

        step_resp = requests.post(
            f"{APP_URL}/api/step",
            json={"session_id": session_id, "action": action},
            timeout=30,
        )
        step_resp.raise_for_status()
        result = step_resp.json()

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        total_reward += reward
        steps_taken += 1

        log_step(obs["step"], action, reward, done, obs)

    # Grade
    grade_resp = requests.get(
        f"{APP_URL}/api/grade/{session_id}",
        timeout=30,
    )
    grade_resp.raise_for_status()
    score = grade_resp.json()["score"]

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
        print("Make sure the server is running (npm run dev) or APP_URL is set correctly.", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task in tasks:
        score = run_task(task)
        scores[task["id"]] = score

    print("\n--- SUMMARY ---", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.4f}", flush=True)
