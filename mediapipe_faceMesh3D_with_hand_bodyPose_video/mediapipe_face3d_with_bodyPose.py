
#v2 face mesh 3d with body pose

# =============================================================================
# MediaPipe offers open source cross-platform, customizable ML solutions for live and streaming media.
#  End-to-End acceleration: Built-in fast ML inference and processing accelerated even on common hardware. 
#  Build once, deploy anywhere: Unified solution works across Android, iOS, desktop/cloud, web and IoT.
# 
# =============================================================================


import mediapipe as mp
import cv2
import datetime
import imutils

mp_holistic = mp.solutions.holistic
# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils 
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#video =r"C:\Users\hp\Downloads\Video\Supported Mini-Squats.mp4"

video =r"C:\Users\hp\Downloads\Video\videoplayback.mp4"
cap = cv2.VideoCapture(video)
#cap =cv2.VideoCapture(0)

#FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
# Initialize MediaPipe Holistic.
with mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ret, image =cap.read()
        
        if ret:
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            image = imutils.resize(image, width=1200)
            total_frames = total_frames + 1
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print nose coordinates.
            image_hight, image_width, _ = image.shape
            if results.pose_landmarks:
              print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_hight})'
              )
            # Draw pose landmarks.
           
            annotated_image = image.copy()
            mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                image=annotated_image, 
                landmark_list=results.face_landmarks, 
                connections=mp_holistic.FACE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1))
            mp_drawing.draw_landmarks(
                image=annotated_image, 
                landmark_list=results.pose_landmarks, 
                connections=mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
            
            
            #FPS
            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)
        
            fps_text = "FPS: {:.2f}".format(fps)
        
            cv2.putText(annotated_image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
         
            #show images
            cv2.imshow("annotated_image", annotated_image)


            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            break
        
cap.release()
cv2.destroyAllWindows()









# =============================================================================
# 
# #boody tracking with face mesh
# 
# import cv2
# import mediapipe as mp
# import imutils
# import datetime
# 
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
# 
# # For webcam input:
# video =r"C:\Users\hp\Downloads\Video\Supported Mini-Squats.mp4"
# 
# #video =r"C:\Users\hp\Downloads\Video\videoplayback.mp4"
# cap = cv2.VideoCapture(video)
# #cap =cv2.VideoCapture(0)
# 
# #FPS
# fps_start_time = datetime.datetime.now()
# fps = 0
# total_frames = 0
# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as pose:
#   while cap.isOpened():
#     success, image = cap.read()
#     image = imutils.resize(image, width=1200)
#     total_frames = total_frames + 1
#     
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
# 
#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = pose.process(image)
# 
#     # Draw the pose annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#     
#     
#     #FPS
#     fps_end_time = datetime.datetime.now()
#     time_diff = fps_end_time - fps_start_time
#     if time_diff.seconds == 0:
#         fps = 0.0
#     else:
#         fps = (total_frames / time_diff.seconds)
# 
#     fps_text = "FPS: {:.2f}".format(fps)
# 
#     cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
#  
#     cv2.imshow('MediaPipe Pose', image)
#     # press "Q" to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 
# cap.release()
# cv2.destroyAllWindows()
# 
# =============================================================================
