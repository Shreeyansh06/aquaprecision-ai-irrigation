# AquaPrecision: AI Irrigation Simulator
# Hugging Face Spaces deployment
 
FROM node:20-slim
 
WORKDIR /app
 
# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci
 
# Copy all source files
COPY . .
 
# Build only the Vite frontend
RUN npx vite build
 
# HF Spaces runs on port 7860
ENV PORT=7860
ENV NODE_ENV=production
 
EXPOSE 7860
 
# Run server directly with tsx (avoids bundling issues)
CMD ["npx", "tsx", "server.ts"]
