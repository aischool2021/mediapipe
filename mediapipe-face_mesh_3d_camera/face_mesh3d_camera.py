
#face mesh 3d

#video
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(80,110,10))
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

cap=cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=3,
    min_detection_confidence=0.5) as face_mesh:
    while True:
        ret, image=cap.read()
          
        if ret:
            results = face_mesh.process(image)
        
            if results.multi_face_landmarks:
                
                for face_landmarks in results.multi_face_landmarks:
                    
                    mp_drawing.draw_landmarks(
                      image=image,
                      landmark_list=face_landmarks,
                      connections=mp_face_mesh.FACE_CONNECTIONS,
                      landmark_drawing_spec=drawing_spec,
                      connection_drawing_spec=drawing_spec)
              
            cv2.imshow("annotated_image", image)
        
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            break
        
cap.release()
cv2.destroyAllWindows()


