🏥 Clinical Documentation System

Offline NLP-Based Nurse Documentation Assistant

The Clinical Documentation System is an offline-first, nurse-focused documentation assistant. It accepts natural language input, extracts structured clinical information, maintains a chronological patient timeline, and generates clean, nursing-style notes.

⚠️ The system is strictly limited to documentation tasks only.
It does not provide diagnoses, treatment plans, or medical recommendations.

📋 Prerequisites
📂 Project Structure (Important)

Backend → FastAPI (Python)

The backend code is inside the app folder

Move the backend out of the app folder before running, otherwise it will not work correctly

Frontend → Next.js / React app

⚠️ Do not run backend and frontend in the same terminal
Each must run in a separate terminal window

Backend Requirements

Python 3.8 or higher

pip

Internet connection (only required once for initial model download)

Frontend Requirements

Node.js 18 or higher

npm / yarn / pnpm / bun

🚀 Backend Setup (FastAPI)
1. Create and Activate a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

2. Install Python Dependencies
pip install fastapi uvicorn transformers torch faiss-cpu numpy

3. Download AI Models (Run Once)
python download.py


This step downloads all required NLP models.
After this, the system can run fully offline.

4. Start the Backend Server
python api_server.py

🌐 Frontend Setup (Next.js)
5. Start the Frontend Server
npm run dev

▶️ Running Summary

Terminal 1 → Backend (FastAPI)

Terminal 2 → Frontend (Next.js)

✔ Backend must be running before using the frontend
✔ System works fully offline after the initial model download