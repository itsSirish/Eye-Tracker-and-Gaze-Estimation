import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import random

i = 0
p1_fitness = 0
p2_fitness = 0
c1_fitness = 0
c2_fitness = 0
num_points = 12
chromosome = [0] * 12
current = [0] * 12
parent1 = [0] * 12
parent2 = [0] * 12
no_of_children = 3
child1 = [0] * 12
child2 = [0] * 12

points = [252, 253, 259, 256, 362, 398, 382, 384, 381, 386, 260, 466, 249,
          390, 373, 374, 263, 388, 387, 380, 385, 258, 257, 254]

expected_values = [[0.5, 0.5], [0, 0.5], [1, 0.5]]

var = ['1', '2', '3']
num_images = len(var)

for top in range(4):

    for x in range(12):
        random_index = random.randrange(0, 24)
        chromosome[x] = points[random_index]

    print("The selected points are")
    print(chromosome)
    fitness = 0  # declaration of fitness

    for image in range(num_images):
        img = cv2.imread('C:\\Users\\Prasanna\\Desktop\\custom dataset\\' + var[image] + '.jpg')
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = img.shape
        screen_w, screen_h = pyautogui.size()

        if landmark_points:
            landmarks = landmark_points[0].landmark
            x_pupil = landmarks[473].x
            y_pupil = landmarks[473].y
            print("pupil co-ordinates ", x_pupil, y_pupil)

            right = [landmarks[473]]
            for landmark in right:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(img, (x, y), 3, (255, 0, 0))

            for feature in range(num_points):
                x = int(landmarks[chromosome[feature]].x * frame_w)
                y = int(landmarks[chromosome[feature]].y * frame_h)
                current[feature] = chromosome[feature]
                cv2.circle(img, (x, y), 3, (255, 0, 0))
            print(current)

            x_min = (landmarks[chromosome[0]].x + landmarks[chromosome[1]].x + landmarks[chromosome[2]].x) / 3
            x_max = (landmarks[chromosome[3]].x + landmarks[chromosome[4]].x + landmarks[chromosome[5]].x) / 3
            y_min = (landmarks[chromosome[6]].y + landmarks[chromosome[7]].y + landmarks[chromosome[8]].y) / 3
            y_max = (landmarks[chromosome[9]].y + landmarks[chromosome[10]].y + landmarks[chromosome[11]].y) / 3
            # print("x min ", x_min)
            # print("x max ", x_max)
            # print("y min ", y_min)
            # print("y max ", y_max)

            eye_width = abs(x_max - x_min)
            eye_height = abs(y_max - y_min)

            x_cord_pred = abs(x_pupil - x_min) / eye_width
            y_cord_pred = abs(y_pupil - y_min) / eye_height

            screen_x = screen_w * x_cord_pred
            screen_y = screen_h * y_cord_pred

            x_actual = expected_values[image][0]
            y_actual = expected_values[image][1]

            error_x = abs(x_actual - x_cord_pred) * 100  # / x_actual  #fitness function
            error_y = abs(y_actual - y_cord_pred) * 100  # / y_actual stopped cause div by 0

            fitness = fitness + 100 - (error_x + error_y) / 2

            print("for image ", image + 1, "error values in run ", top + 1, " are")
            print(error_x, error_y)
            print("total fitness value is at ", fitness)
            print(" ")

        cv2.imshow('Eye Controlled Mouse', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows();

    fitness = fitness / num_images

    if fitness > p1_fitness and fitness > p2_fitness:
        for feature in range(num_points):
            parent2[feature] = parent1[feature]
        p2_fitness = p1_fitness
        for feature in range(num_points):
            parent1[feature] = current[feature]
        p1_fitness = fitness

    elif p1_fitness > fitness > p2_fitness:
        for feature in range(num_points):
            parent2[feature] = current[feature]
        p2_fitness = fitness

    print("the fitness of gen 1 is ", p1_fitness, p2_fitness)
    print(" ");

print("The parent with most fitness", parent1)
print("The parent with second most fitness", parent2)

# childrens run

for num in range(no_of_children):
    random_index = random.randrange(0, 11)
    print("random index is ", random_index)
    for child_index in range(12):
        if child_index < random_index:
            current[child_index] = parent1[child_index]
        if child_index >= random_index:
            current[child_index] = parent2[child_index]
    print("child number", num + 1, "is", current)
    fitness = 0
    for image in range(num_images):
        img = cv2.imread('C:\\Users\\Prasanna\\Desktop\\custom dataset\\' + var[image] + '.jpg')
        face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = face_mesh.process(rgb_frame)
        landmark_points = output.multi_face_landmarks
        frame_h, frame_w, _ = img.shape
        screen_w, screen_h = pyautogui.size()

        if landmark_points:
            landmarks = landmark_points[0].landmark
            x_pupil = landmarks[473].x
            y_pupil = landmarks[473].y
            print("pupil co-ordinates ", x_pupil, y_pupil)

            right = [landmarks[473]]
            for landmark in right:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(img, (x, y), 3, (255, 0, 0))

            for feature in range(num_points):
                x = int(landmarks[current[feature]].x * frame_w)
                y = int(landmarks[current[feature]].y * frame_h)
                cv2.circle(img, (x, y), 3, (255, 0, 0))
            print(current)

            x_min = (landmarks[current[0]].x + landmarks[current[1]].x + landmarks[current[2]].x) / 3
            x_max = (landmarks[current[3]].x + landmarks[current[4]].x + landmarks[current[5]].x) / 3
            y_min = (landmarks[current[6]].y + landmarks[current[7]].y + landmarks[current[8]].y) / 3
            y_max = (landmarks[current[9]].y + landmarks[current[10]].y + landmarks[current[11]].y) / 3

            eye_width = abs(x_max - x_min)
            eye_height = abs(y_max - y_min)

            x_cord_pred = abs(x_pupil - x_min) / eye_width
            y_cord_pred = abs(y_pupil - y_min) / eye_height

            screen_x = screen_w * x_cord_pred
            screen_y = screen_h * y_cord_pred

            x_actual = expected_values[image][0]
            y_actual = expected_values[image][1]

            error_x = abs(x_actual - x_cord_pred) * 100  # / x_actual  #fitness function
            error_y = abs(y_actual - y_cord_pred) * 100  # / y_actual stopped cause div by 0

            fitness = fitness + 100 - (error_x + error_y) / 2

            print("for image ", image + 1, "error values in run ", top + 1, " are")
            print(error_x, error_y)
            print("total fitness value is at ", fitness)
            print(" ")

        cv2.imshow('Eye Controlled Mouse', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows();

    fitness = fitness / num_images

    if fitness > c1_fitness and fitness > c2_fitness:
        for feature in range(num_points):
            child2[feature] = child1[feature]
        c2_fitness = c1_fitness
        for feature in range(num_points):
            child1[feature] = current[feature]
        c1_fitness = fitness

    elif c1_fitness > fitness > c2_fitness:
        for feature in range(num_points):
            child2[feature] = current[feature]
        c2_fitness = fitness
    print("The child with most fitness", child1)
    print("The child with second most fitness", child2)

    print("the fitness of gen 2 is ", c1_fitness, c2_fitness)
    print(" ");

print("The child with most fitness", child1)
print("The child with second most fitness", child2)

