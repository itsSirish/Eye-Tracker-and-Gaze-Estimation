import cv2
import mediapipe as mp
import pyautogui
cam = cv2.VideoCapture(0)
ret, im = cam.read()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    if landmark_points:
        landmarks = landmark_points[0].landmark

        right = [landmarks[362], landmarks[263], landmarks[386], landmarks[253], landmarks[473]]

        x_min = (landmarks[253].x + landmarks[253].x + landmarks[362].x)/3
        x_pupil = landmarks[473].x
        x_max = (landmarks[254].x + landmarks[387].x + landmarks[252].x)/3
        eye_width = abs(x_max - x_min)

        y_min = landmarks[374].y
        y_pupil = landmarks[473].y
        y_max = landmarks[386].y
        eye_height = abs(y_max - y_min)

        #print(y_min, y_pupil, y_max, eye_height)

        x_cord = abs(x_pupil - x_min) / eye_width
        y_cord = abs(y_pupil - y_min) / eye_height

        screen_x = screen_w * x_cord
        screen_y = screen_h * y_cord

        print(x_pupil, y_pupil)

        pyautogui.moveTo(screen_x, screen_y)



    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)