# AquaPrecision: AI Irrigation Simulator
# Hugging Face Spaces deployment

FROM node:20-slim


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN pip3 install openai requests --break-system-packages

WORKDIR /app


COPY package*.json ./
RUN npm ci


COPY . .


RUN npx vite build


ENV PORT=7860
ENV NODE_ENV=production

EXPOSE 7860


CMD ["npx", "tsx", "server.ts"]
