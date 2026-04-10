
"""
AquaPrecision: AI Irrigation Simulator — OpenEnv Inference Script
Follows the EXACT stdout format required by the hackathon judges.
"""

import asyncio
import os
import sys
import json
import requests
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_URL = os.getenv("APP_URL", os.getenv("SPACE_URL", "http://localhost:7860")).rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = HF_TOKEN or OPENAI_API_KEY or "dummy"

BENCHMARK = "aquaprecision-irrigation"
SUCCESS_SCORE_THRESHOLD = 0.1

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


# ---------------------------------------------------------------------------
# ✅ EXACT log format required by judges
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Score clamping — strictly between 0 and 1
# ---------------------------------------------------------------------------

def clamp_score(value) -> float:
    try:
        v = float(value)
        if v != v:
            v = 0.5
    except (TypeError, ValueError):
        v = 0.5
    v = max(0.001, min(0.999, v))
    return float(round(v, 4))


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
    if driest.get("moisture", 1) < 0.4 and float(water.get("current", 0)) >= 5:
        return {"type": "irrigate", "cell_id": driest["id"], "amount": 5}
    return {"type": "wait"}


# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

def llm_action(obs: dict, task_description: str) -> dict:
    field = obs.get("field", [])
    weather = obs.get("weather", {})
    water = obs.get("water_tank", {})

    prompt = (
        f"Task: {task_description}\n"
        f"Step: {obs.get('step', 0)}\n"
        f"Water tank: {water.get('current', 0):.1f} / {water.get('capacity', 100):.1f} L\n"
        f"Field cells:\n"
        + "\n".join(
            f"  cell {c['id']}: moisture={c.get('moisture', 0):.2f}, "
            f"health={c.get('crop_health', 0):.2f}, dead={c.get('is_dead', False)}"
            for c in field
        )
        + "\n\nGoal: Maximise crop health while conserving water.\n"
        "Respond with ONLY JSON:\n"
        '  {"type": "irrigate", "cell_id": <int>, "amount": <float 1-10>}\n'
        '  or {"type": "wait"}'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI irrigation agent. Return only valid JSON."},
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
            raise ValueError(f"Bad action type: {action}")
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return heuristic_action(obs)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict) -> float:
    task_name = task.get("name", task["id"])
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.5
    success = False

    try:
        # Reset
        reset_resp = requests.post(
            f"{APP_URL}/api/reset",
            json={"task_id": task["id"], "seed": "openenv_inference_seed_42"},
            timeout=30,
        )
        reset_resp.raise_for_status()
        data = reset_resp.json()
        session_id = data["session_id"]
        obs = data["observation"]
        done = False

        while not done:
            action = llm_action(obs, task.get("description", task["id"]))
            action_str = json.dumps(action)

            try:
                step_resp = requests.post(
                    f"{APP_URL}/api/step",
                    json={"session_id": session_id, "action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                result = step_resp.json()
            except Exception as exc:
                log_step(steps_taken + 1, action_str, 0.0, True, str(exc))
                break

            obs = result["observation"]
            reward = float(clamp_score(result.get("reward", 0.5)))
            done = result["done"]
            steps_taken += 1
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=None
            )

        # Grade
        try:
            grade_resp = requests.get(f"{APP_URL}/api/grade/{session_id}", timeout=30)
            grade_resp.raise_for_status()
            raw_score = grade_resp.json().get("score", 0.5)
            score = float(clamp_score(raw_score))
        except Exception as exc:
            print(f"[DEBUG] Grading error: {exc}", file=sys.stderr, flush=True)
            score = float(clamp_score(
                sum(rewards) / max(len(rewards), 1)
            ))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task error: {exc}", file=sys.stderr, flush=True)
        score = 0.5
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Entry point — EXACT format required by judges
# ---------------------------------------------------------------------------

async def main() -> None:
    try:
        tasks_resp = requests.get(f"{APP_URL}/api/tasks", timeout=30)
        tasks_resp.raise_for_status()
        tasks = tasks_resp.json()
    except Exception as exc:
        print(f"[DEBUG] ERROR: Could not fetch tasks — {exc}", file=sys.stderr)
        sys.exit(1)

    for task in tasks:
        try:
            run_task(task)
        except Exception as exc:
            print(f"[DEBUG] ERROR running {task['id']}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
 flush=True)
