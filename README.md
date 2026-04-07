---
title: AquaPrecision AI Irrigation Simulator
emoji: 🌱
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# AquaPrecision: AI Irrigation Simulator (OpenEnv)

A real-world simulation environment for AI agents to manage precision irrigation in drought-prone regions. This environment follows the **OpenEnv** specification, providing a standard interface for reinforcement learning and agent evaluation.

## Environment Description

The simulation models a crop field in an arid region. An AI agent must decide when and where to irrigate to maximize crop health while minimizing water waste.

### Observation Space
- **Step**: Current time step.
- **Weather**: Temperature, Humidity, Rain status, Evaporation rate.
- **Field**: Grid of cells with moisture levels, crop health, and status (alive/dead).
- **Water Tank**: Current water level and total capacity.

### Action Space
- `irrigate(cell_id, amount)`: Apply `amount` liters of water to a specific cell.
- `wait()`: Do nothing for one step.

### Reward Function
- `+ Health Bonus`: Proportional to average crop health.
- `- Water Penalty`: Proportional to water used.
- `- Death Penalty`: Large penalty for each crop that dies.

## Tasks

1. **Survival Basics (Easy)**: 1x1 grid, 10 steps. Learn to keep a single plant alive.
2. **Field Management (Medium)**: 3x3 grid, 20 steps. Manage multiple plants with moderate weather changes.
3. **Drought Crisis (Hard)**: 5x5 grid, 50 steps. Extreme heat and very limited water resources.

## Setup & Usage

### Local Development
1. Install dependencies: `npm install`
2. Start the server: `npm run dev`
3. Open `http://localhost:3000` to see the visualization.

### OpenEnv API
The environment exposes a REST API:
- `GET /api/tasks`: List available tasks.
- `POST /api/reset`: Reset environment with `task_id`. Returns `session_id` and initial observation.
- `POST /api/step`: Take an action. Requires `session_id` and `action` object.
- `GET /api/grade/:session_id`: Get the final performance score (0.0 - 1.0).

## Baseline Inference Script

A Python baseline script is provided in `baseline_inference.py`. It uses a simple heuristic to manage the field.

```bash
# Example usage
export APP_URL="http://localhost:3000"
python baseline_inference.py
```

## Deployment

The environment is containerized and can be deployed to Hugging Face Spaces or any Docker-compatible host.

```bash
docker build -t aquaprecision .
docker run -p 3000:3000 aquaprecision
```
