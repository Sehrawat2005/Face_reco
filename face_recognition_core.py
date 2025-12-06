import cv2
import numpy as np
from deepface import DeepFace
import os
import json
from datetime import datetime, timedelta

class FaceAttendance:
    def __init__(self, known_faces_dir="known_faces", cooldown_minutes=5):
        self.known_faces_dir = known_faces_dir
        self.cooldown_minutes = cooldown_minutes
        self.known_encodings = {}
        self.last_logged = {}  # name: datetime of last log
        self.load_known_faces()
    
    def load_known_faces(self):
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            return
        
        for person_folder in os.listdir(self.known_faces_dir):
            person_path = os.path.join(self.known_faces_dir, person_folder)
            if os.path.isdir(person_path):
                embeddings = []
                for img_file in os.listdir(person_path)[:5]:
                    img_path = os.path.join(person_path, img_file)
                    try:
                        embedding = DeepFace.represent(img_path, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
                        norm_embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(np.array(norm_embedding))
                    except:
                        continue
                if embeddings:
                    self.known_encodings[person_folder] = embeddings
        print(f"Loaded {len(self.known_encodings)} known faces")
    
    def recognize_face(self, face_img):
        try:
            embedding = DeepFace.represent(face_img, model_name='VGG-Face', enforce_detection=False)[0]['embedding']
            norm_embedding = embedding / np.linalg.norm(embedding)
            
            best_match = None
            best_distance = float('inf')
            
            for name, encodings in self.known_encodings.items():
                for known_encoding in encodings:
                    distance = np.linalg.norm(norm_embedding - known_encoding)
                    if distance < best_distance and distance < 0.7:
                        best_distance = distance
                        best_match = name
            
            return best_match
        except:
            return None
    
    def mark_attendance(self, name):
        now = datetime.now()
        if name in self.last_logged:
            elapsed = now - self.last_logged[name]
            if elapsed < timedelta(minutes=self.cooldown_minutes):
                return None
        
        self.last_logged[name] = now
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        record = {"name": name, "time": timestamp}
        
        attendance_file = "attendance_records.json"
        if os.path.exists(attendance_file):
            with open(attendance_file, 'r') as f:
                records = json.load(f)
        else:
            records = []
        
        records.append(record)
        with open(attendance_file, 'w') as f:
            json.dump(records, f, indent=2)
        
        return timestamp
