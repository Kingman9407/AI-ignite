"""
process.py
Handles deterministic text generation for nursing notes.
NO extraction logic. Only language generation.
"""

import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class NursingNoteGenerator:
    """Generates nursing documentation notes using FLAN-T5-Small."""
    
    def __init__(self):
        """Load FLAN-T5-Small model from local cache."""
        model_path = os.path.join("models", "flan_t5_small")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please run download.py first."
            )
        
        print("Loading FLAN-T5-Small model...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        self.model.eval()
        print("✓ Model loaded successfully\n")
    
    def _generate_deterministic(self, prompt):
        """
        Generate text with deterministic settings.
        No randomness, no interpretation.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,
                do_sample=False,  # Deterministic
                temperature=1.0,  # Not used when do_sample=False
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_nursing_note(self, symptom_events):
        """
        Generate a nursing-style symptom documentation note.
        
        Args:
            symptom_events: List of symptom event dictionaries
        
        Returns:
            str: Professional nursing documentation
        """
        if not symptom_events:
            return "No symptoms documented during this period."
        
        # Build structured symptom data string
        symptom_data = []
        for event in symptom_events:
            symptom = event.get("symptom", "unknown")
            time = event.get("time_of_day", "unspecified time")
            food = event.get("relation_to_food", "no food relation noted")
            freq = event.get("frequency_marker", "")
            
            entry = f"{symptom} at {time}"
            if food and food != "no food relation noted":
                entry += f" {food}"
            if freq:
                entry += f" ({freq})"
            
            symptom_data.append(entry)
        
        # Create prompt for FLAN-T5
        prompt = (
            "Write a nursing documentation note based on these reported symptoms. "
            "Do not diagnose or interpret. Only document what was reported. "
            "Use professional nursing language. "
            f"Symptoms: {', '.join(symptom_data)}. "
            "Documentation note:"
        )
        
        note = self._generate_deterministic(prompt)
        
        # Add documentation header
        header = "=== SYMPTOM DOCUMENTATION ===\n"
        footer = "\n\nNote: This is documentation only. No clinical interpretation provided."
        
        return header + note + footer
    
    def generate_medication_note(self, medication_events):
        """
        Generate a Medication Administration Record (MAR) style note.
        
        Args:
            medication_events: List of medication event dictionaries
        
        Returns:
            str: Professional medication documentation
        """
        if not medication_events:
            return "No medications documented during this period."
        
        # Build structured medication data string
        med_data = []
        for event in medication_events:
            med = event.get("medication", "unknown medication")
            dose = event.get("dose", "dose not specified")
            time = event.get("time_of_day", "unspecified time")
            food = event.get("relation_to_food", "")
            route = event.get("route", "oral")
            
            entry = f"{med} {dose} via {route} at {time}"
            if food:
                entry += f" {food}"
            
            med_data.append(entry)
        
        # Create prompt for FLAN-T5
        prompt = (
            "Write a medication administration record note. "
            "Do not provide dosage advice or recommendations. "
            "Only document what was taken as reported. "
            "Use professional nursing language. "
            f"Medications: {', '.join(med_data)}. "
            "MAR note:"
        )
        
        note = self._generate_deterministic(prompt)
        
        # Add documentation header
        header = "=== MEDICATION ADMINISTRATION RECORD ===\n"
        footer = "\n\nNote: Patient-reported intake. Documentation only."
        
        return header + note + footer
    
    def generate_timeline_summary(self, all_events):
        """
        Generate a chronological summary of all events.
        
        Args:
            all_events: List of all events (symptoms + medications)
        
        Returns:
            str: Chronological documentation summary
        """
        if not all_events:
            return "No events documented."
        
        # Sort by timestamp
        sorted_events = sorted(all_events, key=lambda x: x.get("timestamp", ""))
        
        # Build event descriptions
        event_descriptions = []
        for event in sorted_events:
            if "symptom" in event:
                desc = f"Symptom: {event['symptom']}"
            elif "medication" in event:
                desc = f"Medication: {event['medication']} {event.get('dose', '')}"
            else:
                continue
            
            desc += f" at {event.get('time_of_day', 'unspecified time')}"
            event_descriptions.append(desc)
        
        # Create prompt
        prompt = (
            "Write a brief chronological nursing summary of these documented events. "
            "Do not interpret or diagnose. Only summarize what was documented. "
            f"Events: {'; '.join(event_descriptions)}. "
            "Summary:"
        )
        
        summary = self._generate_deterministic(prompt)
        
        return "=== CHRONOLOGICAL SUMMARY ===\n" + summary