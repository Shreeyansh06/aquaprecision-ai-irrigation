import os
import requests
import json
from openai import OpenAI

# Configuration
APP_URL = os.getenv("APP_URL", os.getenv("SPACE_URL", "http://localhost:3000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment. Agent will use fallback heuristic.")

client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

def get_agent_action(obs, task_description):
    """Uses OpenAI to decide the next action based on observation."""
    if not OPENAI_API_KEY:
        # Fallback to heuristic if no key
        alive_cells = [c for c in obs["field"] if not c["is_dead"]]
        if not alive_cells: return {"type": "wait"}
        driest = min(alive_cells, key=lambda x: x["moisture"])
        if driest["moisture"] < 0.4:
            return {"type": "irrigate", "cell_id": driest["id"], "amount": 5}
        return {"type": "wait"}

    prompt = f"""
Task: {task_description}
Current Observation:
- Step: {obs['step']}
- Weather: {obs['weather']}
- Water Tank: {obs['water_tank']}
- Field: {obs['field']}

Goal: Maximize crop health and survival while being efficient with water.
Decide the next action. You can 'irrigate' a specific cell (amount 0-10) or 'wait'.
Return ONLY a JSON object in this format: {{"type": "irrigate", "cell_id": 0, "amount": 5}} or {{"type": "wait"}}
"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert AI irrigation agent managing a drought-prone farm."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" },
            temperature=0 # For reproducibility
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Agent Error: {e}")
        return {"type": "wait"}

def run_task(task):
    print(f"\n{'='*60}")
    print(f"RUNNING TASK: {task['name']} ({task['difficulty']})")
    print(f"Description: {task['description']}")
    print(f"{'='*60}")
    
    # 1. Reset with a fixed seed for reproducibility
    seed = "openenv_baseline_seed_42"
    res = requests.post(f"{APP_URL}/api/reset", json={
        "task_id": task["id"],
        "seed": seed
    })
    data = res.json()
    session_id = data["session_id"]
    obs = data["observation"]
    
    done = False
    total_reward = 0
    
    while not done:
        action = get_agent_action(obs, task["description"])
        
        res = requests.post(f"{APP_URL}/api/step", json={
            "session_id": session_id,
            "action": action
        })
        result = res.json()
        
        obs = result["observation"]
        total_reward += result["reward"]
        done = result["done"]
        
        action_str = f"IRRIGATE {action.get('amount')}L on {action.get('cell_id')}" if action['type'] == 'irrigate' else "WAIT"
        print(f"Step {obs['step']:02d} | Action: {action_str:15} | Reward: {result['reward']:6.2f} | Water: {obs['water_tank']['current']:.1f}L")

    # 4. Final Grade
    res = requests.get(f"{APP_URL}/api/grade/{session_id}")
    grade = res.json()["score"]
    print(f"{'-'*60}")
    print(f"TASK COMPLETE. FINAL SCORE: {grade:.4f}")
    print(f"{'='*60}\n")
    return grade

if __name__ == "__main__":
    try:
        tasks = requests.get(f"{APP_URL}/api/tasks").json()
        scores = {}
        for task in tasks:
            score = run_task(task)
            scores[task['name']] = score
            
        print("\nSUMMARY OF BASELINE SCORES:")
        for name, score in scores.items():
            print(f"- {name:20}: {score:.4f}")
            
    except Exception as e:
        print(f"Error connecting to environment: {e}")
        print("Make sure the dev server is running at http://localhost:3000")
