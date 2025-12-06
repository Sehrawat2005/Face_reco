import cv2
from face_recognition_core import FaceAttendance
from deepface import DeepFace

attendance = FaceAttendance()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("=== FACE ATTENDANCE SYSTEM ===")
print("Press 'q' to quit")
print("Add faces: known_faces/YourName/ (3-5 photos)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    status_text = "No Face Detected"
    color = (0, 0, 255)  # Red
    
    try:
        faces = DeepFace.extract_faces(frame, enforce_detection=False)
        
        if faces is None:
            faces = []
        
        print(f"Faces detected: {len(faces)}")
        
        # FIXED: ONLY show "Face Detected" if faces > 0
        if len(faces) > 0:
            status_text = "Face Detected"
            color = (0, 255, 0)  # Green
            
            for face_data in faces:
                if 'facial_area' not in face_data or face_data['facial_area'] is None:
                    continue
                    
                facial_area = face_data['facial_area']
                x = facial_area.get('x', 0)
                y = facial_area.get('y', 0)
                w = facial_area.get('w', 50)
                h = facial_area.get('h', 50)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
                
                if 'face' in face_data and face_data['face'] is not None:
                    face_img = face_data['face']
                    name = attendance.recognize_face(face_img)
                    
                    if name:
                        cv2.putText(frame, f"ID: {name}", (x+5, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        timestamp = attendance.mark_attendance(name)
                        if timestamp:
                            cv2.putText(frame, f"Logged: {timestamp[-8:]}", (x+5, y+h+25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Logged recently", (x+5, y+h+25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x+5, y-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No clear face", (x+5, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Status bar (CORRECT LOGIC)
        cv2.rectangle(frame, (10, 10), (400, 60), color, -1)
        cv2.putText(frame, status_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    except Exception as e:
        print(f"Error: {e}")
        cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 255), -1)
        cv2.putText(frame, "No Face Detected", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow('Face Attendance System [q=Quit]', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Attendance saved to attendance_records.json")
