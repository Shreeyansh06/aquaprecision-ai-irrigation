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

// ✅ Helper: clamp strictly between 0 and 1 (never 0.0 or 1.0)
function clamp(value: number): number {
  return Math.min(0.99, Math.max(0.01, parseFloat(value.toFixed(4))));
}

export function gradeTask(state: State, task: Task): number {
  // ✅ Never return 0 even at step 0
  if (state.step === 0) return 0.01;

  const aliveCrops = state.field.filter(c => !c.is_dead).length;
  const totalCrops = state.field.length;

  // ✅ survivalRate never exactly 0 or 1
  const survivalRate = clamp(aliveCrops / Math.max(totalCrops, 1));

  // ✅ avgHealth never exactly 0 or 1
  const avgHealth = clamp(
    state.field.reduce((acc, c) => acc + c.crop_health, 0) / Math.max(totalCrops, 1)
  );

  // ✅ waterEfficiency never exactly 0 or 1
  const waterEfficiency = clamp(
    state.water_tank / Math.max(task.config.initial_water, 1)
  );

  // Score components
  const survivalScore = survivalRate * 0.6;
  const healthScore = avgHealth * 0.3;
  const efficiencyScore = waterEfficiency * 0.1;

  const finalScore = survivalScore + healthScore + efficiencyScore;

  // ✅ Final score strictly between 0 and 1
  return clamp(finalScore);
}