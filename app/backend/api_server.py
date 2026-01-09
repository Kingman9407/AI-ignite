"""
api_server.py
FastAPI server - bridge to clinical documentation system.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime
import sys
import traceback

app = FastAPI(title="Clinical Documentation API")

# ==============================
# CORS CONFIG
# ==============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# SYSTEM INIT
# ==============================

clinical_system = None
system_error = None

@app.on_event("startup")
async def startup_event():
    global clinical_system, system_error
    try:
        print("Initializing Clinical Documentation System...")
        
        # Import here to catch import errors
        from main import ClinicalDocumentationSystem
        
        # Initialize with error handling
        clinical_system = ClinicalDocumentationSystem()
        print("✓ Clinical Documentation System initialized successfully")
        
    except Exception as e:
        system_error = str(e)
        print(f"✗ Failed to initialize system: {e}")
        print(traceback.format_exc())
        print("\n" + "=" * 60)
        print("WARNING: Running in fallback mode (no ML models)")
        print("=" * 60)
        
        # Create a minimal fallback system
        clinical_system = create_fallback_system()

# ==============================
# FALLBACK SYSTEM (No ML)
# ==============================

class FallbackSystem:
    """Minimal system without ML models for testing."""
    
    def __init__(self):
        self.patient_state = {
            "patient_info": {"age": 62, "gender": "male"},
            "symptom_events": [],
            "medication_events": []
        }
        
        self.symptoms = [
            "headache", "chest pain", "nausea", "dizziness", "fatigue",
            "fever", "breathlessness", "cough", "sore throat", "vomiting",
            "abdominal pain", "back pain", "joint pain", "muscle pain"
        ]
        
        self.medications = [
            "metformin", "aspirin", "paracetamol", "ibuprofen", "lisinopril",
            "amlodipine", "omeprazole", "levothyroxine", "atorvastatin"
        ]
    
    def _extract_symptom(self, text):
        text_lower = text.lower()
        for symptom in self.symptoms:
            if symptom in text_lower:
                return symptom
        return None
    
    def _extract_medication(self, text):
        text_lower = text.lower()
        for med in self.medications:
            if med in text_lower:
                return med
        return None
    
    def _extract_time(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in ["morning", "am", "breakfast"]):
            return "morning"
        if any(w in text_lower for w in ["afternoon", "lunch", "noon"]):
            return "afternoon"
        if any(w in text_lower for w in ["evening", "dinner"]):
            return "evening"
        if any(w in text_lower for w in ["night", "bedtime", "pm"]):
            return "night"
        return "unspecified"

def create_fallback_system():
    """Create fallback system without ML models."""
    return FallbackSystem()

# ==============================
# MODELS
# ==============================

class ChatRequest(BaseModel):
    text: str | None = None
    message: str | None = None

# ==============================
# HELPER FUNCTIONS
# ==============================

def process_chat_input(text: str) -> str:
    """Process user input and return formatted response."""
    try:
        text_lower = text.lower().strip()
        
        # Check for symptom
        symptom = clinical_system._extract_symptom(text)
        medication = clinical_system._extract_medication(text)
        
        if symptom:
            return process_symptom(text, symptom)
        elif medication:
            return process_medication(text, medication)
        elif "timeline" in text_lower:
            return get_timeline()
        elif "help" in text_lower:
            return get_help_message()
        else:
            return get_help_message()
            
    except Exception as e:
        print(f"Error in process_chat_input: {e}")
        traceback.print_exc()
        return f"Error processing input: {str(e)}"

def process_symptom(text: str, symptom: str) -> str:
    """Process symptom documentation."""
    try:
        time_of_day = "unspecified"
        if hasattr(clinical_system, '_extract_time'):
            time_of_day = clinical_system._extract_time(text)
        
        event = {
            "symptom": symptom,
            "time_of_day": time_of_day,
            "timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        clinical_system.patient_state["symptom_events"].append(event)
        
        response = f"✅ Symptom documented: {symptom}\n"
        response += f"   Time: {time_of_day}\n"
        response += f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}"
        
        return response
        
    except Exception as e:
        print(f"Error in process_symptom: {e}")
        return f"Error documenting symptom: {str(e)}"

def process_medication(text: str, medication: str) -> str:
    """Process medication documentation."""
    try:
        time_of_day = "unspecified"
        if hasattr(clinical_system, '_extract_time'):
            time_of_day = clinical_system._extract_time(text)
        
        event = {
            "medication": medication,
            "time_of_day": time_of_day,
            "timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        clinical_system.patient_state["medication_events"].append(event)
        
        response = f"✅ Medication documented: {medication}\n"
        response += f"   Time: {time_of_day}\n"
        response += f"   Timestamp: {datetime.now().strftime('%H:%M:%S')}"
        
        return response
        
    except Exception as e:
        print(f"Error in process_medication: {e}")
        return f"Error documenting medication: {str(e)}"

def get_timeline() -> str:
    """Get formatted timeline of events."""
    try:
        all_events = []
        
        for event in clinical_system.patient_state["symptom_events"]:
            all_events.append({
                "type": "SYMPTOM",
                "timestamp": event["timestamp"],
                "description": f"{event['symptom']} at {event.get('time_of_day', 'unknown time')}"
            })
        
        for event in clinical_system.patient_state["medication_events"]:
            all_events.append({
                "type": "MEDICATION",
                "timestamp": event["timestamp"],
                "description": f"{event['medication']} at {event.get('time_of_day', 'unknown time')}"
            })
        
        all_events.sort(key=lambda x: x["timestamp"])
        
        if not all_events:
            return "📋 No events documented yet.\n\nTry: 'Patient has headache after breakfast'"
        
        response = "📋 PATIENT TIMELINE\n" + "=" * 40 + "\n\n"
        
        for event in all_events:
            ts = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M")
            response += f"[{ts}] {event['type']}\n"
            response += f"  {event['description']}\n\n"
        
        return response
        
    except Exception as e:
        print(f"Error in get_timeline: {e}")
        return f"Error getting timeline: {str(e)}"

def get_help_message() -> str:
    """Return help message."""
    return """🏥 Clinical Documentation Assistant

I can help you document:

📝 Symptoms:
   "Patient has headache after breakfast"
   "Chest pain in the evening"

💊 Medications:
   "Gave metformin 1000mg in morning"
   "Lisinopril after dinner"

📊 View data:
   "Show timeline"

Try describing a symptom or medication!"""

# ==============================
# ROUTES
# ==============================

@app.get("/api/patient-info")
async def get_patient_info():
    """Get patient information."""
    try:
        if not clinical_system:
            return {
                "success": False,
                "error": "System not initialized",
                "patient_info": {"age": 0, "gender": "unknown"},
                "symptom_count": 0,
                "medication_count": 0
            }

        state = clinical_system.patient_state
        patient_info = state.get("patient_info", {"age": 0, "gender": "unknown"})
        symptom_events = state.get("symptom_events", [])
        medication_events = state.get("medication_events", [])

        return {
            "success": True,
            "patient_info": patient_info,
            "symptom_count": len(symptom_events),
            "medication_count": len(medication_events),
            "system_mode": "fallback" if isinstance(clinical_system, FallbackSystem) else "full"
        }

    except Exception as e:
        print(f"Error in get_patient_info: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "patient_info": {"age": 0, "gender": "unknown"},
            "symptom_count": 0,
            "medication_count": 0
        }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Process user input through clinical documentation system."""
    try:
        user_text = request.text or request.message

        if not user_text:
            raise HTTPException(status_code=400, detail="No text provided")

        if not clinical_system:
            raise HTTPException(status_code=500, detail="Clinical system not initialized")

        response = process_chat_input(user_text)
        
        return {
            "reply": response,
            "success": True
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return {
            "reply": f"Error: {str(e)}",
            "success": False
        }

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLINICAL DOCUMENTATION CHAT BRIDGE")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("Chat endpoint: POST http://localhost:8000/api/chat")
    print("Patient info: GET http://localhost:8000/api/patient-info")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)