import { Task, State } from './types';

export const TASKS: Task[] = [
  {
    id: 'task_easy_1',
    name: 'Survival Basics',
    description: 'Maintain a single crop cell for 10 steps. Water is plentiful.',
    difficulty: 'easy',
    max_steps: 10,
    config: {
      grid_size: 1,
      initial_water: 100,
      evaporation_multiplier: 1.0,
      weather_volatility: 0.2,
    },
  },
  {
    id: 'task_medium_1',
    name: 'Field Management',
    description: 'Manage a 3x3 grid for 20 steps with moderate weather volatility.',
    difficulty: 'medium',
    max_steps: 20,
    config: {
      grid_size: 3,
      initial_water: 200,
      evaporation_multiplier: 1.2,
      weather_volatility: 0.5,
    },
  },
  {
    id: 'task_hard_1',
    name: 'Drought Crisis',
    description: 'A 5x5 grid, extreme heat, and very limited water. Survive for 50 steps.',
    difficulty: 'hard',
    max_steps: 50,
    config: {
      grid_size: 5,
      initial_water: 150,
      evaporation_multiplier: 1.5,
      weather_volatility: 0.8,
    },
  },
];

// ✅ Strictly between 0 and 1 — never 0.0 or 1.0
function clamp(value: number): number {
  const v = isNaN(value) || !isFinite(value) ? 0.5 : value;
  return parseFloat(Math.min(0.95, Math.max(0.02, v)).toFixed(4));
}

export function gradeTask(state: State, task: Task): number {
  if (!state || !state.field || state.field.length === 0) return 0.02;
  if (state.step === 0) return 0.02;

  const totalCrops = state.field.length;

  // ✅ alive count — at least 1 to avoid 0
  const aliveCrops = Math.max(1, state.field.filter(c => !c.is_dead).length);
  const survivalRate = clamp(aliveCrops / totalCrops);

  // ✅ health — floor at 0.02
  const rawHealth = state.field.reduce((acc, c) => acc + Math.max(0.02, c.crop_health || 0.02), 0) / totalCrops;
  const avgHealth = clamp(rawHealth);

  // ✅ water — never 0, never equal to initial
  const rawWater = Math.max(1, state.water_tank || 1);
  const initialWater = Math.max(1, task.config.initial_water || 1);
  const waterEfficiency = clamp(rawWater / initialWater);

  // Weighted score
  const finalScore = (survivalRate * 0.5) + (avgHealth * 0.3) + (waterEfficiency * 0.2);

  return clamp(finalScore);
}
