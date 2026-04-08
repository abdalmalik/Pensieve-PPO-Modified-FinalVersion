# Pensieve WebApp GitHub Ready

This folder contains a clean GitHub-ready copy of the `webapp` part of the project. It keeps the Flask backend, the React/Vite frontend, and the Replit helpers, while excluding generated folders such as `node_modules`, `dist`, cache files, and local test logs.

## What This App Does

The web app provides an interactive Adaptive Bitrate Streaming demo with two controller modes:

- `simulated`: fast rule-based behavior
- `real`: loads a real Pensieve PPO checkpoint and applies the safe-step inference logic

## Important Note About Models

This web app supports both of these layouts:

1. Inside the same repository, for example `repo-root/webapp/` with the Pensieve code in `repo-root/src/`
2. Beside a separate clean Pensieve project copy on the Desktop, for example:

```text
Desktop/
├─ Pensieve-PPO-GitHub-Ready/
└─ Pensieve-WebApp-GitHub-Ready/
```

In either layout, the app can automatically discover checkpoints such as `src/ppo/nn_model_ep_500000.pth`.

## Included Structure

```text
Pensieve-WebApp-GitHub-Ready/
├─ agent_runtime.py
├─ server.py
├─ run_replit.py
├─ requirements.txt
├─ .replit
├─ replit.nix
├─ static/
├─ templates/
└─ frontend/
   ├─ public/
   ├─ src/
   ├─ index.html
   ├─ package.json
   ├─ package-lock.json
   └─ vite.config.js
```

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer
- npm

Python dependencies:

- `Flask`
- `numpy`
- `torch`

Frontend dependencies are installed from `frontend/package.json`.

## Local Run

Install the backend dependency:

```bash
pip install -r requirements.txt
```

Install the frontend dependencies and build the app:

```bash
cd frontend
npm install
npm run build
cd ..
```

Start the server:

```bash
python server.py
```

Then open:

```text
http://localhost:8000
```

## Development Mode

Run Flask:

```bash
python server.py
```

In a second terminal run Vite:

```bash
cd frontend
npm install
npm run dev
```

During development, Vite runs on port `5173` and proxies API calls to Flask on port `8000`.

## Replit Run

You can also run:

```bash
python run_replit.py
```

This script installs Python packages when needed, installs frontend packages when needed, builds the frontend, and launches the Flask server.

## Notes

- `frontend/node_modules` and `frontend/dist` are intentionally excluded from this copy.
- For real PPO inference, keep either `webapp/` inside the same repository that contains `src/ppo2.py` and the `.pth` checkpoints, or place it beside a companion Pensieve project copy.
- The backend already includes the improved safe-step inference behavior used in the latest evaluation.
