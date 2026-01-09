"""
main.py
Interactive CLI nurse documentation system.
Offline, deterministic, append-only timeline.
"""

import os
import re
from datetime import datetime
from collections import defaultdict
import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from process import NursingNoteGenerator

class ClinicalDocumentationSystem:
    """Offline nurse-style clinical documentation system."""
    
    def __init__(self):
        """Initialize the documentation system."""
        # Patient state structure
        self.patient_state = {
    "patient_info": {
        "age": 62,
        "gender": "male"
    },
    "symptom_events": [
        {
            "symptom": "headache",
            "time_of_day": "morning",
            "relation_to_food": "after food",
            "frequency_marker": "again",
            "timestamp": "2026-01-08T09:30"
        },
        {
            "symptom": "chest pain",
            "time_of_day": "night",
            "relation_to_food": "after food",
            "frequency_marker": None,
            "timestamp": "2026-01-07T21:15"
        }
    ],
    "medication_events": [
        {
            "medication": "metformin",
            "dose": "1000 mg",
            "time_of_day": "morning",
            "relation_to_food": "after food",
            "route": "oral",
            "timestamp": "2026-01-08T08:00",
            "note": "patient reported intake"
        },
        {
            "medication": "lisinopril",
            "dose": None,
            "time_of_day": "night",
            "relation_to_food": "before food",
            "route": "oral",
            "timestamp": "2026-01-07T19:30",
            "note": "patient reported intake"
        }
    ]
}

        
        # Load BioClinicalBERT for embeddings
        self._load_embeddings_model()
        
        # Initialize FAISS index
        self.embedding_dim = 768  # BioClinicalBERT dimension
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.event_store = []  # Parallel store for events
        
        # Load text generation model
        self.note_generator = NursingNoteGenerator()
        
        # Symptom vocabulary (safe list)
        self.symptoms = [
            "headache", "chest pain", "nausea", "dizziness", "fatigue",
            "fever", "breathlessness", "cough", "sore throat", "vomiting",
            "abdominal pain", "back pain", "joint pain", "muscle pain",
            "shortness of breath", "palpitations", "sweating", "chills"
        ]
        
        # Medication vocabulary (safe list)
        self.medications = [
            "metformin", "aspirin", "paracetamol", "ibuprofen", "lisinopril",
            "amlodipine", "omeprazole", "levothyroxine", "atorvastatin",
            "losartan", "metoprolol", "albuterol", "insulin", "warfarin",
            "clopidogrel", "prednisone", "amoxicillin", "azithromycin"
        ]
        
        # Time patterns
        self.time_patterns = {
            "morning": ["morning", "am", "breakfast"],
            "afternoon": ["afternoon", "lunch", "noon"],
            "evening": ["evening", "dinner"],
            "night": ["night", "bedtime", "pm"]
        }
        
        # Food relation patterns
        self.food_patterns = {
            "after food": ["after food", "after eating", "after meal", "after breakfast", "after lunch", "after dinner"],
            "before food": ["before food", "before eating", "before meal", "before breakfast", "before lunch", "before dinner"],
            "with food": ["with food", "with meal", "with meals", "during meal"],
            "empty stomach": ["empty stomach", "on empty stomach", "without food"]
        }
        
        # Frequency markers
        self.frequency_markers = ["again", "twice", "three times", "multiple times", "every day", "daily", "since yesterday"]
    
    def _load_embeddings_model(self):
        """Load BioClinicalBERT for semantic embeddings."""
        model_path = os.path.join("models", "bio_clinical_bert")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run download.py first."
            )
        
        print("Loading BioClinicalBERT model...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.bert_model = AutoModel.from_pretrained(model_path, local_files_only=True)
        self.bert_model.eval()
        print("✓ BioClinicalBERT loaded successfully")
    
    def _get_embedding(self, text):
        """Generate embedding for text using BioClinicalBERT."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding[0]
    
    def _extract_symptom(self, text):
        """Extract symptom from text using rule-based matching."""
        text_lower = text.lower()
        
        for symptom in self.symptoms:
            if symptom in text_lower:
                return symptom
        
        return None
    
    def _extract_medication(self, text):
        """Extract medication name from text."""
        text_lower = text.lower()
        
        for med in self.medications:
            if med in text_lower:
                return med
        
        return None
    
    def _extract_dose(self, text):
        """Extract medication dose from text."""
        # Pattern: number followed by mg/tablet/units
        dose_patterns = [
            r'(\d+\.?\d*\s*mg)',
            r'(\d+\s*tablet[s]?)',
            r'(\d+\s*unit[s]?)',
            r'(\d+\.?\d*\s*ml)'
        ]
        
        for pattern in dose_patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _extract_time_of_day(self, text):
        """Extract time of day from text."""
        text_lower = text.lower()
        
        # Check for exact time (e.g., "8 am", "9:30 pm")
        time_match = re.search(r'(\d{1,2}):?(\d{2})?\s*(am|pm)', text_lower)
        if time_match:
            return time_match.group(0)
        
        # Check for general time of day
        for time_key, patterns in self.time_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return time_key
        
        return "unspecified time"
    
    def _extract_food_relation(self, text):
        """Extract relation to food from text."""
        text_lower = text.lower()
        
        for relation, patterns in self.food_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return relation
        
        return None
    
    def _extract_frequency_marker(self, text):
        """Extract frequency marker from text."""
        text_lower = text.lower()
        
        for marker in self.frequency_markers:
            if marker in text_lower:
                return marker
        
        return None
    
    def record_symptom(self, text):
        """Record a symptom event."""
        symptom = self._extract_symptom(text)
        
        if not symptom:
            print("✗ No recognized symptom found in text.")
            print(f"  Recognized symptoms: {', '.join(self.symptoms[:10])}...")
            return
        
        # Extract contextual information
        time_of_day = self._extract_time_of_day(text)
        relation_to_food = self._extract_food_relation(text)
        frequency_marker = self._extract_frequency_marker(text)
        
        # Create symptom event
        event = {
            "symptom": symptom,
            "time_of_day": time_of_day,
            "relation_to_food": relation_to_food,
            "frequency_marker": frequency_marker,
            "timestamp": datetime.now().isoformat(),
            "raw_text": text
        }
        
        # Append to timeline
        self.patient_state["symptom_events"].append(event)
        
        # Add to FAISS index
        embedding = self._get_embedding(f"symptom {symptom} {time_of_day}")
        self.faiss_index.add(np.array([embedding], dtype=np.float32))
        self.event_store.append({"type": "symptom", "data": event})
        
        print(f"✓ Symptom documented: {symptom}")
        print(f"  Time: {time_of_day}")
        if relation_to_food:
            print(f"  Food relation: {relation_to_food}")
        if frequency_marker:
            print(f"  Frequency: {frequency_marker}")
    
    def record_medication(self, text):
        """Record a medication event."""
        medication = self._extract_medication(text)
        
        if not medication:
            print("✗ No recognized medication found in text.")
            print(f"  Recognized medications: {', '.join(self.medications[:10])}...")
            return
        
        # Extract contextual information
        dose = self._extract_dose(text)
        time_of_day = self._extract_time_of_day(text)
        relation_to_food = self._extract_food_relation(text)
        
        # Create medication event
        event = {
            "medication": medication,
            "dose": dose if dose else "dose not specified",
            "time_of_day": time_of_day,
            "relation_to_food": relation_to_food,
            "route": "oral",  # Default
            "timestamp": datetime.now().isoformat(),
            "note": "patient reported intake",
            "raw_text": text
        }
        
        # Append to timeline
        self.patient_state["medication_events"].append(event)
        
        # Add to FAISS index
        embedding = self._get_embedding(f"medication {medication} {dose}")
        self.faiss_index.add(np.array([embedding], dtype=np.float32))
        self.event_store.append({"type": "medication", "data": event})
        
        print(f"✓ Medication documented: {medication}")
        print(f"  Dose: {dose if dose else 'not specified'}")
        print(f"  Time: {time_of_day}")
        if relation_to_food:
            print(f"  Food relation: {relation_to_food}")
    
    def show_timeline(self):
        """Display chronological timeline of all events."""
        all_events = []
        
        for event in self.patient_state["symptom_events"]:
            all_events.append({
                "type": "SYMPTOM",
                "timestamp": event["timestamp"],
                "description": f"{event['symptom']} at {event['time_of_day']}",
                "details": event
            })
        
        for event in self.patient_state["medication_events"]:
            all_events.append({
                "type": "MEDICATION",
                "timestamp": event["timestamp"],
                "description": f"{event['medication']} {event['dose']} at {event['time_of_day']}",
                "details": event
            })
        
        # Sort by timestamp
        all_events.sort(key=lambda x: x["timestamp"])
        
        if not all_events:
            print("\nNo events documented yet.")
            return
        
        print("\n" + "=" * 60)
        print("PATIENT TIMELINE")
        print("=" * 60)
        
        for event in all_events:
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}] {event['type']}")
            print(f"  {event['description']}")
            
            if event["type"] == "SYMPTOM":
                details = event["details"]
                if details.get("relation_to_food"):
                    print(f"  Food relation: {details['relation_to_food']}")
                if details.get("frequency_marker"):
                    print(f"  Frequency: {details['frequency_marker']}")
            
            elif event["type"] == "MEDICATION":
                details = event["details"]
                if details.get("relation_to_food"):
                    print(f"  Food relation: {details['relation_to_food']}")
                print(f"  Route: {details['route']}")
        
        print("\n" + "=" * 60)
    
    def symptom_frequency(self, symptom):
        """Show frequency of a specific symptom."""
        symptom_lower = symptom.lower()
        
        matching_events = [
            event for event in self.patient_state["symptom_events"]
            if event["symptom"].lower() == symptom_lower
        ]
        
        if not matching_events:
            print(f"\n✗ No documented instances of '{symptom}'")
            return
        
        print(f"\n{'=' * 60}")
        print(f"FREQUENCY REPORT: {symptom.upper()}")
        print(f"{'=' * 60}")
        print(f"Total occurrences: {len(matching_events)}")
        print(f"\nDetailed records:")
        
        for i, event in enumerate(matching_events, 1):
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n  {i}. [{timestamp}]")
            print(f"     Time of day: {event['time_of_day']}")
            if event.get("relation_to_food"):
                print(f"     Food relation: {event['relation_to_food']}")
            if event.get("frequency_marker"):
                print(f"     Frequency marker: {event['frequency_marker']}")
        
        print(f"\n{'=' * 60}")
    
    def medication_frequency(self, medication):
        """Show frequency of a specific medication."""
        med_lower = medication.lower()
        
        matching_events = [
            event for event in self.patient_state["medication_events"]
            if event["medication"].lower() == med_lower
        ]
        
        if not matching_events:
            print(f"\n✗ No documented instances of '{medication}'")
            return
        
        print(f"\n{'=' * 60}")
        print(f"MEDICATION FREQUENCY REPORT: {medication.upper()}")
        print(f"{'=' * 60}")
        print(f"Total administrations: {len(matching_events)}")
        print(f"\nDetailed records:")
        
        for i, event in enumerate(matching_events, 1):
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n  {i}. [{timestamp}]")
            print(f"     Dose: {event['dose']}")
            print(f"     Time of day: {event['time_of_day']}")
            if event.get("relation_to_food"):
                print(f"     Food relation: {event['relation_to_food']}")
            print(f"     Route: {event['route']}")
        
        print(f"\n{'=' * 60}")
    
    def generate_nursing_note(self):
        """Generate nursing documentation note for symptoms."""
        if not self.patient_state["symptom_events"]:
            print("\nNo symptoms documented. Cannot generate note.")
            return
        
        print("\nGenerating nursing note...\n")
        note = self.note_generator.generate_nursing_note(
            self.patient_state["symptom_events"]
        )
        print(note)
    
    def generate_medication_note(self):
        """Generate medication administration record note."""
        if not self.patient_state["medication_events"]:
            print("\nNo medications documented. Cannot generate note.")
            return
        
        print("\nGenerating medication note...\n")
        note = self.note_generator.generate_medication_note(
            self.patient_state["medication_events"]
        )
        print(note)
    
    def run(self):
        """Run the interactive CLI."""
        print("\n" + "=" * 60)
        print("CLINICAL DOCUMENTATION SYSTEM")
        print("Offline Nurse Documentation Interface")
        print("=" * 60)
        print("\nCommands:")
        print("  note <text>          - Record symptom")
        print("  med <text>           - Record medication")
        print("  timeline             - View all events")
        print("  frequency <symptom>  - View symptom frequency")
        print("  med_frequency <med>  - View medication frequency")
        print("  nurse_note           - Generate nursing note")
        print("  med_note             - Generate medication note")
        print("  help                 - Show commands")
        print("  exit                 - Exit system")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "exit":
                    print("\nExiting documentation system.")
                    break
                
                elif user_input.lower() == "help":
                    print("\nCommands:")
                    print("  note <text>          - Record symptom")
                    print("  med <text>           - Record medication")
                    print("  timeline             - View all events")
                    print("  frequency <symptom>  - View symptom frequency")
                    print("  med_frequency <med>  - View medication frequency")
                    print("  nurse_note           - Generate nursing note")
                    print("  med_note             - Generate medication note")
                
                elif user_input.lower().startswith("note "):
                    text = user_input[5:].strip()
                    self.record_symptom(text)
                
                elif user_input.lower().startswith("med "):
                    text = user_input[4:].strip()
                    self.record_medication(text)
                
                elif user_input.lower() == "timeline":
                    self.show_timeline()
                
                elif user_input.lower().startswith("frequency "):
                    symptom = user_input[10:].strip()
                    self.symptom_frequency(symptom)
                
                elif user_input.lower().startswith("med_frequency "):
                    medication = user_input[14:].strip()
                    self.medication_frequency(medication)
                
                elif user_input.lower() == "nurse_note":
                    self.generate_nursing_note()
                
                elif user_input.lower() == "med_note":
                    self.generate_medication_note()
                
                else:
                    print("✗ Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\n\nExiting documentation system.")
                break
            except Exception as e:
                print(f"✗ Error: {e}")

def main():
    """Main entry point."""
    try:
        system = ClinicalDocumentationSystem()
        system.run()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease run download.py first to download required models.")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")

if __name__ == "__main__":
    main()