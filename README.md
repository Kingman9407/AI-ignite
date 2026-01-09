# Clinical Documentation System  
Offline NLP-Based Nurse Documentation Assistant

The Clinical Documentation System is an offline-first nursing documentation assistant that accepts natural language input, extracts structured clinical information, maintains a chronological patient timeline, and generates clean nursing-style notes.  
The system is strictly limited to documentation tasks and does not provide diagnosis or treatment recommendations.

---

## Project Structure (Important)

Backend and frontend must be run separately.

Backend → FastAPI (Python)  
NOTE THE BACKEND IS IN THE APP FOLDER HERE FOR SINGLE DEPLOY MOVE  OUTSIDE TO WORK PROPERLY
Frontend → Next.js / React

Do not run backend and frontend in the same terminal.  
Each must run in its own terminal window.

---

## Requirements

### Backend
- Python 3.8+
- pip
- Internet connection required only for first-time model download

### Frontend
- Node.js 18+
- npm / yarn / pnpm / bun

---

## Backend Setup (FastAPI)

### 1. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate      # Windows
```

### 2. Install Python dependencies

```bash
pip install fastapi uvicorn transformers torch faiss-cpu numpy
```

### 3. Download models (run once)

```bash
python download.py
```

### 4. Start backend server

```bash
python api_server.py
```

---

## Frontend Setup

### Start Next.js dev server

```bash
npm run dev
```

---

## Running Summary

Terminal 1 → Backend (FastAPI)  
Terminal 2 → Frontend (Next.js)

Backend must be running first.  
System works offline after first model download.
