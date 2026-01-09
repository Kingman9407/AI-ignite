# 🏥 Clinical Documentation System  
Offline NLP-Based Nurse Documentation Assistant

The Clinical Documentation System is an offline-first nursing documentation assistant that accepts natural language input, extracts structured clinical information, maintains a chronological patient timeline, and generates clean nursing-style notes. The system is strictly limited to documentation tasks and does not provide diagnosis or treatment recommendations.

---

## 📋 Prerequisites
📂 Project Structure (Important)

The backend and frontend must be run separately.

Backend → FastAPI (Python)
the back end is in the app folder move out to run properly

Frontend → Next.js / React app

⚠️ Do not run backend and frontend in the same terminal.
Each must run in its own terminal window.

### Backend
- Python 3.8 or higher
- pip
- Internet connection (required only for initial model download)

### Frontend
- Node.js 18 or higher
- npm / yarn / pnpm / bun

---

## 🚀 Backend Setup (FastAPI)

### 1. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/binactivate      # macOS / Linux
# venv\Scripts\activate      # Windows

2. Install Python Dependencies

pip install fastapi uvicorn transformers torch faiss-cpu numpy

3. Download AI Models (Run Once)

python download.py

5. Start the Backend Server

python api_server.py

4. Start the Frontend Server

npm run dev


Running Summary

Terminal 1 → Backend (FastAPI)

Terminal 2 → Frontend (Next.js)

Backend must be running before using the frontend

System works fully offline after initial model download
