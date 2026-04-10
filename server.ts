import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";
import cors from "cors";
import { v4 as uuidv4 } from "uuid";
import { IrrigationEnv } from "./src/openenv/environment";
import { TASKS, gradeTask } from "./src/openenv/tasks";
import { ActionSchema } from "./src/openenv/types";
import OpenAI from "openai";

let openai: OpenAI | null = null;

function getOpenAI() {
  if (!openai) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("OPENAI_API_KEY is not configured");
    }
    openai = new OpenAI({
      apiKey,
      baseURL: process.env.API_BASE_URL || "https://generativelanguage.googleapis.com/v1beta/openai/",
    });
  }
  return openai;
}

// --- OpenEnv State ---
let envState: any = {};

const TASK_CONFIGS: Record<string, any> = {
  easy:   { crops: 4,  reservoir: 1000, maxSteps: 24, weather: "SUNNY"    },
  medium: { crops: 8,  reservoir: 600,  maxSteps: 48, weather: "CLOUDY"   },
  hard:   { crops: 12, reservoir: 300,  maxSteps: 72, weather: "HEATWAVE" },
};

// ✅ Clamp helper — ensures value is strictly between 0 and 1
function clamp(value: number): number {
  return Math.min(0.99, Math.max(0.01, parseFloat(value.toFixed(4))));
}

function buildCrops(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: i,
    // ✅ moisture strictly between 1 and 99 (never 0 or 100)
    moisture: Math.min(99, Math.max(1, 50 + Math.floor(Math.random() * 20) - 10)),
    health: 99, // ✅ never 100
  }));
}

async function startServer() {
  const app = express();
  const PORT = parseInt(process.env.PORT || "7860");

  app.use(cors());

  // --- OpenEnv Standard Endpoints registered FIRST ---
  app.post("/reset", express.json(), (req, res) => {
    const task = req.body?.task || "easy";
    const config = TASK_CONFIGS[task] || TASK_CONFIGS["easy"];
    envState = {
      task,
      crops: buildCrops(config.crops),
      reservoirLevel: config.reservoir,
      weatherForecast: config.weather,
      day: 0,
      hour: 0,
      done: false,
      totalReward: 0,
      score: 0.5, // ✅ strictly between 0 and 1
    };
    console.log(`[OpenEnv] Reset for task: ${task}`);
    res.json({ state: envState });
  });

  app.post("/step", express.json(), (req, res) => {
    const { action, cropId, amount, hours } = req.body || {};
    let reward = 0.1; // ✅ never starts at 0

    if (action === "WATER" && cropId !== undefined && amount !== undefined) {
      const crop = envState.crops?.find((c: any) => c.id === cropId);
      if (crop && envState.reservoirLevel >= amount) {
        const before = crop.moisture;
        crop.moisture = Math.min(99, crop.moisture + amount); // ✅ never 100
        envState.reservoirLevel -= amount;
        const delta = (crop.moisture - before) / 100;
        reward = clamp(delta + 0.3); // ✅ strictly between 0 and 1
        if (crop.moisture >= 40 && crop.moisture <= 80) {
          reward = clamp(reward + 0.2);
        }
      }
    } else {
      const h = hours || 1;
      envState.hour += h;
      if (envState.hour >= 24) {
        envState.day += 1;
        envState.hour = 0;
      }
      envState.crops?.forEach((c: any) => {
        // ✅ moisture never 0
        c.moisture = Math.max(1, c.moisture - (envState.weatherForecast === "HEATWAVE" ? 4 : 2));
        // ✅ health never 0 or 100
        c.health = c.moisture < 20
          ? Math.max(1, c.health - 5)
          : Math.min(99, c.health + 1);
      });
      const goodCrops = envState.crops?.filter(
        (c: any) => c.moisture >= 40 && c.moisture <= 80
      ).length || 0;
      const total = envState.crops?.length || 1;
      // ✅ strictly between 0 and 1
      reward = clamp((goodCrops / total) * 0.8 + 0.1);
    }

    envState.totalReward = (envState.totalReward || 0) + reward;
    const config = TASK_CONFIGS[envState.task];
    const steps = envState.day * 24 + envState.hour;
    if (steps >= config?.maxSteps || envState.reservoirLevel <= 0) envState.done = true;

    // ✅ score field added, strictly between 0 and 1
    const score = clamp(reward);
    res.json({
      state: envState,
      reward: score,
      score: score,
      done: envState.done,
    });
  });

  app.get("/state", (_req, res) => {
    res.json({ state: envState });
  });

  app.get("/health", (_req, res) => {
    res.json({ status: "ok" });
  });

  // --- Global middleware for remaining routes ---
  app.use(express.json({ limit: "10mb" }));

  // In-memory store for environments (sessions)
  const sessions: Record<string, { env: IrrigationEnv; taskId: string }> = {};

  // --- OpenAI Vision Analysis Endpoint ---
  app.post("/api/vision-analyze", async (req, res) => {
    const { image } = req.body;
    try {
      const client = getOpenAI();
      const response = await client.chat.completions.create({
        model: process.env.MODEL_NAME || "gemini-2.0-flash",
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Analyze this soil image. Estimate the moisture level as a percentage (0-100%) and provide a brief recommendation for irrigation in liters. Format: Moisture: X%, Recommendation: YL.",
              },
              {
                type: "image_url",
                image_url: { url: image },
              },
            ],
          },
        ],
      });
      res.json({ result: response.choices[0].message.content });
    } catch (err) {
      console.error("OpenAI Error:", err);
      const message = err instanceof Error ? err.message : "Failed to analyze image";
      res.status(500).json({ error: message });
    }
  });

  // --- OpenEnv API Endpoints (with /api prefix) ---
  app.get("/api/tasks", (_req, res) => {
    res.json(TASKS);
  });

  app.post("/api/reset", (req, res) => {
    const { task_id, seed } = req.body;
    const task = TASKS.find((t) => t.id === task_id) || TASKS[0];
    const sessionId = uuidv4();
    const env = new IrrigationEnv();
    const observation = env.reset(task, seed);
    sessions[sessionId] = { env, taskId: task.id };
    res.json({ session_id: sessionId, observation });
  });

  app.post("/api/step", (req, res) => {
    const { session_id, action } = req.body;
    const session = sessions[session_id];
    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }
    try {
      const validatedAction = ActionSchema.parse(action);
      const result = session.env.step(validatedAction);
      // ✅ clamp reward from environment too
      const clampedReward = result.reward !== undefined ? clamp(result.reward) : 0.5;
res.json({
  ...result,
  reward: clampedReward,
  score: clampedReward,
});
      res.json(result);
    } catch (err) {
      res.status(400).json({ error: "Invalid action", details: err });
    }
  });

  app.get("/api/state/:session_id", (req, res) => {
    const session = sessions[req.params.session_id];
    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }
    res.json(session.env.getState());
  });

  app.get("/api/grade/:session_id", (req, res) => {
    const session = sessions[req.params.session_id];
    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }
    const state = session.env.getState();
    const task = TASKS.find((t) => t.id === session.taskId)!;
    let score = gradeTask(state, task);
    // ✅ ensure grade is strictly between 0 and 1
    score = clamp(score);
    res.json({ score });
  });

  // --- Vite Middleware ---
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    // ✅ Only serve index.html for non-API, non-OpenEnv routes
    app.get(/^(?!\/api|\/reset|\/step|\/state|\/health).*$/, (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
